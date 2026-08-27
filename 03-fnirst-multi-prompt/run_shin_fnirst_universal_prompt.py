# -*- coding: utf-8 -*-
"""Train fNIRS-T single or weighted multi-prompts with a universal EEG adapter."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader

from fnirst_universal_prompt_model import (
    FrozenCBraModFNIRSTPrompt,
    WeightedMultiPrompt,
    official_all_patch_classifier,
)


PROJECT_DIR = Path(__file__).resolve().parent
from shin_multimodal_data import (
    SHINTrialDataset,
    TASKS,
    fit_train_fnirs_stats,
    load_split,
)


BRANCHES = ("eeg", "fnirs", "fusion")
TRAINED_BRANCHES = ("fnirs", "fusion")
EEG_SCALE_DIVISOR = 100.0
PROMPT_STEPS = {
    "single": 1,
    "multi_weighted": 2,
    "learned_context_sample": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    return parser.parse_args()


def read_config(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    required = {
        "task", "prompt_mode", "eeg_root", "fnirs_root", "cbramod_root",
        "checkpoint", "reference_head_checkpoint", "cache_dir", "seed",
        "epochs", "batch_size", "prompt_lr", "adapter_lr", "fnirs_head_lr",
        "weight_decay", "dropout", "early_stopping_patience", "loss_weights",
        "router_hidden", "router_temperature", "entropy_lambda", "num_workers",
        "device", "deterministic", "cudnn_benchmark",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing config fields: {missing}")
    if config["task"] not in TASKS:
        raise ValueError(f"Unknown task: {config['task']}")
    if config["prompt_mode"] not in PROMPT_STEPS:
        raise ValueError(
            "prompt_mode must be single, multi_weighted, or "
            "learned_context_sample"
        )
    config.setdefault("context_tokens", 8)
    config.setdefault("fnirs_epoch_start_s", 0.0)
    config.setdefault("fnirs_epoch_stop_s", 10.0)
    config.setdefault("eeg_epoch_start_s", 0.0)
    config.setdefault("eeg_epoch_stop_s", 10.0)
    if int(config["context_tokens"]) < 1:
        raise ValueError("context_tokens must be positive")
    fnirs_duration = float(config["fnirs_epoch_stop_s"]) - float(
        config["fnirs_epoch_start_s"]
    )
    eeg_duration = float(config["eeg_epoch_stop_s"]) - float(
        config["eeg_epoch_start_s"]
    )
    if fnirs_duration <= 0 or eeg_duration <= 0:
        raise ValueError("EEG/fNIRS epoch durations must be positive")
    if not math.isclose(eeg_duration, 10.0):
        raise ValueError("The fixed CBraMod reference requires exactly 10 s EEG")
    fnirs_sampling_points = int(round(fnirs_duration * 10.0))
    if not math.isclose(fnirs_sampling_points / 10.0, fnirs_duration):
        raise ValueError("fNIRS epoch must align to its 10 Hz sampling grid")
    config["fnirs_sampling_points"] = fnirs_sampling_points
    if set(config["loss_weights"]) != set(TRAINED_BRANCHES):
        raise ValueError(f"loss_weights must contain exactly {TRAINED_BRANCHES}")
    if int(config["epochs"]) != 50 and not args.smoke:
        raise ValueError("The controlled experiment uses 50 epochs")
    if config["task"] == "mi" and config["early_stopping_patience"] is not None:
        raise ValueError("MI uses 50 epochs without early stopping")
    if config["task"] == "ma" and int(config["early_stopping_patience"] or 0) != 15:
        raise ValueError("MA uses early-stopping patience 15")
    if float(config["router_temperature"]) <= 0:
        raise ValueError("router_temperature must be positive")
    if float(config["entropy_lambda"]) < 0:
        raise ValueError("entropy_lambda must be nonnegative")
    config["output_dir"] = str(args.output_dir.resolve())
    config["config_source"] = str(args.config.resolve())
    config["diagnose_only"] = bool(args.diagnose_only)
    config["smoke"] = bool(args.smoke)
    config["max_train_batches"] = args.max_train_batches
    config["max_subjects_per_split"] = 1 if (args.diagnose_only or args.smoke) else None
    if args.device:
        config["device"] = args.device
    if args.cache_dir:
        config["cache_dir"] = str(args.cache_dir.resolve())
    if args.smoke:
        config["epochs"] = 2
        config["max_train_batches"] = args.max_train_batches or 2
    return config


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def seed_everything(seed: int, deterministic: bool, benchmark: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


def limited_subjects(start: int, stop: int, limit: int | None) -> list[int]:
    subjects = list(range(start, stop + 1))
    return subjects[:limit] if limit else subjects


def load_arrays(config: dict) -> dict:
    task = config["task"]
    limit = config["max_subjects_per_split"]
    split_subjects = {
        "train": limited_subjects(1, 19, limit),
        "val": limited_subjects(20, 24, limit),
        "test": limited_subjects(25, 29, limit),
    }
    arrays = {
        split: load_split(
            eeg_root=Path(config["eeg_root"]),
            fnirs_root=Path(config["fnirs_root"]),
            subjects=subjects,
            split_name=f"fnirst_universal_{task}_{split}",
            task_key=task,
            cache_dir=Path(config["cache_dir"]),
            epoch_start_s=float(config["fnirs_epoch_start_s"]),
            epoch_stop_s=float(config["fnirs_epoch_stop_s"]),
            eeg_epoch_start_s=float(config["eeg_epoch_start_s"]),
            eeg_epoch_stop_s=float(config["eeg_epoch_stop_s"]),
        )
        for split, subjects in split_subjects.items()
    }
    expected = {split: len(subjects) * 60 for split, subjects in split_subjects.items()}
    for split, item in arrays.items():
        if len(item.labels) != expected[split]:
            raise RuntimeError(f"{split}: expected {expected[split]} trials, got {len(item.labels)}")
        if Counter(item.labels.tolist()) != Counter({0: expected[split] // 2, 1: expected[split] // 2}):
            raise RuntimeError(f"{split}: labels are not balanced")
    return arrays


def make_loader(dataset, batch_size, shuffle, workers, seed):
    generator = torch.Generator().manual_seed(seed)

    def worker_init(worker_id: int) -> None:
        value = seed + worker_id
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
        worker_init_fn=worker_init if workers else None,
    )


def prepare_eeg(eeg_uv: torch.Tensor, device: torch.device) -> torch.Tensor:
    eeg = eeg_uv.to(device, non_blocking=True).float() / EEG_SCALE_DIVISOR
    if tuple(eeg.shape[1:]) != (30, 2000):
        raise ValueError(f"Expected EEG [B,30,2000], got {tuple(eeg.shape)}")
    return eeg.reshape(eeg.shape[0], 30, 10, 200)


def load_cbramod(config: dict) -> tuple[nn.Module, dict]:
    root = Path(config["cbramod_root"]).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CBraMod root not found: {root}")
    sys.path.insert(0, str(root))
    from models.cbramod import CBraMod

    backbone = CBraMod(
        in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
        seq_len=30, n_layer=12, nhead=8,
    )
    checkpoint_path = Path(config["checkpoint"]).resolve()
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(source, dict) and "model" in source:
        source = source["model"]
    result = backbone.load_state_dict(source, strict=True)
    backbone.proj_out = nn.Identity()
    return backbone, {
        "checkpoint": str(checkpoint_path),
        "loaded_keys": len(source),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_fixed_classifier(config: dict) -> tuple[nn.Module, dict]:
    classifier = official_all_patch_classifier(dropout=float(config["dropout"]))
    checkpoint_path = Path(config["reference_head_checkpoint"]).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Reference head not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("model", checkpoint)
    prefix = "classifier."
    head_state = {
        name[len(prefix):]: value for name, value in source.items()
        if name.startswith(prefix)
    }
    result = classifier.load_state_dict(head_state, strict=True)
    return classifier, {
        "checkpoint": str(checkpoint_path),
        "sha256": file_sha256(checkpoint_path),
        "loaded_keys": len(head_state),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def build_model(config: dict) -> tuple[FrozenCBraModFNIRSTPrompt, dict]:
    backbone, backbone_record = load_cbramod(config)
    classifier, head_record = load_fixed_classifier(config)
    model = FrozenCBraModFNIRSTPrompt(
        backbone=backbone,
        classifier=classifier,
        prompt_mode=config["prompt_mode"],
        dropout=float(config["dropout"]),
        router_hidden=int(config["router_hidden"]),
        router_temperature=float(config["router_temperature"]),
        entropy_lambda=float(config["entropy_lambda"]),
        context_tokens=int(config["context_tokens"]),
        fnirs_sampling_points=int(config["fnirs_sampling_points"]),
    )
    return model, {"backbone": backbone_record, "reference_head": head_record}


def metric_dict(labels: np.ndarray, predictions: np.ndarray, loss: float) -> dict:
    kappa = float(cohen_kappa_score(labels, predictions))
    if not math.isfinite(kappa):
        kappa = 0.0
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "cohen_kappa": kappa,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def router_statistics(weights: np.ndarray | None) -> dict:
    if weights is None or not len(weights):
        return {}
    effective = 1.0 / np.square(weights).sum(axis=1)
    return {
        "source_names": list(WeightedMultiPrompt.SOURCE_NAMES),
        "mean": weights.mean(axis=0).tolist(),
        "std": weights.std(axis=0).tolist(),
        "min": weights.min(axis=0).tolist(),
        "max": weights.max(axis=0).tolist(),
        "effective_prompt_count_mean": float(effective.mean()),
        "effective_prompt_count_std": float(effective.std()),
    }


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[dict, dict]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    labels, indices = [], []
    logits = {branch: [] for branch in BRANCHES}
    losses = {branch: 0.0 for branch in BRANCHES}
    routes = []
    for eeg, fnirs, target, index in loader:
        target_device = target.to(device, non_blocking=True)
        output = model(
            prepare_eeg(eeg, device),
            fnirs.to(device, non_blocking=True).float(),
        )
        labels.append(target.numpy())
        indices.append(index.numpy())
        if output["router_weights"] is not None:
            routes.append(output["router_weights"].cpu().numpy())
        for branch in BRANCHES:
            losses[branch] += float(criterion(output[branch], target_device).item())
            logits[branch].append(output[branch].cpu().numpy())
    y_true = np.concatenate(labels)
    values: dict[str, Any] = {
        "labels": y_true,
        "indices": np.concatenate(indices),
    }
    metrics = {}
    for branch in BRANCHES:
        branch_logits = np.concatenate(logits[branch])
        prediction = branch_logits.argmax(axis=1)
        probability = torch.softmax(torch.from_numpy(branch_logits), dim=1).numpy()
        metrics[branch] = metric_dict(
            y_true, prediction, losses[branch] / len(y_true)
        )
        values[f"{branch}_predictions"] = prediction
        values[f"{branch}_probabilities"] = probability
    route_array = np.concatenate(routes) if routes else None
    values["router_weights"] = route_array
    metrics["routing"] = router_statistics(route_array)
    metrics["gate"] = model.gate_value()
    return metrics, values


def train_epoch(model, loader, optimizer, device, config):
    model.train()
    criterion = nn.CrossEntropyLoss()
    totals = {"fusion": 0.0, "fnirs": 0.0, "regularization": 0.0, "total": 0.0}
    seen = 0
    for step, (eeg, fnirs, target, _) in enumerate(loader, 1):
        if config["max_train_batches"] and step > int(config["max_train_batches"]):
            break
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            prepare_eeg(eeg, device),
            fnirs.to(device, non_blocking=True).float(),
        )
        fusion_loss = criterion(output["fusion"], target)
        fnirs_loss = criterion(output["fnirs"], target)
        regularization = model.regularization_loss()
        total = (
            float(config["loss_weights"]["fusion"]) * fusion_loss
            + float(config["loss_weights"]["fnirs"]) * fnirs_loss
            + regularization
        )
        total.backward()
        optimizer.step()
        batch = len(target)
        seen += batch
        totals["fusion"] += float(fusion_loss.item()) * batch
        totals["fnirs"] += float(fnirs_loss.item()) * batch
        totals["regularization"] += float(regularization.item()) * batch
        totals["total"] += float(total.item()) * batch
    if not seen:
        raise RuntimeError("No training batches were processed")
    result = {f"{key}_loss": value / seen for key, value in totals.items()}
    result["gate"] = model.gate_value()
    result["routing"] = model.routing_statistics()
    return result


def trainable_state(model) -> dict[str, torch.Tensor]:
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name in trainable_names
    }


def load_trainable_state(model, state) -> None:
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected trainable keys: {result.unexpected_keys}")


def make_optimizer(model, config):
    return torch.optim.AdamW([
        {
            "params": model.prompt_generator.parameters(),
            "lr": float(config["prompt_lr"]),
            "name": "fnirs_t_and_prompt",
        },
        {
            "params": model.adapter.parameters(),
            "lr": float(config["adapter_lr"]),
            "name": "universal_adapter",
        },
        {
            "params": model.fnirs_head.parameters(),
            "lr": float(config["fnirs_head_lr"]),
            "name": "fnirs_head",
        },
    ], weight_decay=float(config["weight_decay"]))


def serializable_values(values: dict) -> dict:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in values.items()
    }


def run_diagnostic(config, datasets, loader, output_dir):
    model, load_record = build_model(config)
    eeg, fnirs, target, _ = next(iter(loader))
    eeg_ready = prepare_eeg(eeg, torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        output = model(eeg_ready, fnirs.float())
    zero_difference = float((output["eeg"] - output["fusion"]).abs().max())
    initial_routes = (
        output["router_weights"].cpu().tolist()
        if output["router_weights"] is not None else None
    )
    with torch.no_grad():
        model.adapter.gate.fill_(0.01)
    model.train()
    opened = model(eeg_ready, fnirs.float())
    loss = (
        nn.functional.cross_entropy(opened["fusion"], target)
        + nn.functional.cross_entropy(opened["fnirs"], target)
        + model.regularization_loss()
    )
    loss.backward()
    missing = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    nonfinite = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
        and not torch.isfinite(parameter.grad).all()
    ]
    diagnostic = {
        "passed": False,
        "task": config["task"],
        "step": PROMPT_STEPS[config["prompt_mode"]],
        "prompt_mode": config["prompt_mode"],
        "architecture": {
            "fnirs_t_input": list(fnirs.shape),
            "eeg_input": list(eeg.shape),
            "prepared_eeg": list(eeg_ready.shape),
            "eeg_tokens": list(output["eeg_tokens"].shape),
            "prompt": list(output["prompt"].shape),
            "eeg_epoch_seconds": [
                config["eeg_epoch_start_s"], config["eeg_epoch_stop_s"]
            ],
            "fnirs_epoch_seconds": [
                config["fnirs_epoch_start_s"], config["fnirs_epoch_stop_s"]
            ],
            "injection": "one final-layer generic residual cross-attention; no spatial/temporal split",
            "zero_gate_exact": zero_difference == 0.0,
            "initial_router_weights": initial_routes,
        },
        "reference": load_record,
        "parameters": {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen_backbone": sum(p.numel() for p in model.backbone.parameters()),
            "frozen_classifier": sum(p.numel() for p in model.classifier.parameters()),
            "fnirs_t": sum(p.numel() for p in model.prompt_generator.fnirs_t.parameters()),
            "adapter": sum(p.numel() for p in model.adapter.parameters()),
        },
        "data": {
            split: {
                "trials": len(dataset),
                "label_counts": dict(Counter(dataset.labels.tolist())),
            }
            for split, dataset in datasets.items()
        },
        "gradient": {
            "loss": float(loss.detach()),
            "missing": missing,
            "nonfinite": nonfinite,
            "passed": not missing and not nonfinite,
        },
    }
    if config["prompt_mode"] == "multi_weighted":
        expected = [1.0 / 3.0] * 3
        route = np.asarray(initial_routes)
        diagnostic["architecture"]["uniform_router_initialization"] = bool(
            np.allclose(route, expected, atol=1e-7)
        )
    diagnostic["passed"] = bool(
        diagnostic["architecture"]["zero_gate_exact"]
        and diagnostic["gradient"]["passed"]
        and (
            config["prompt_mode"] != "multi_weighted"
            or diagnostic["architecture"]["uniform_router_initialization"]
        )
    )
    write_json(output_dir / "diagnostics.json", diagnostic)
    if not diagnostic["passed"]:
        raise RuntimeError(f"Diagnostic failed: {diagnostic}")
    model.to("cpu")
    return diagnostic


def subject_metrics(dataset, values, branch):
    rows = []
    labels = values["labels"]
    predictions = values[f"{branch}_predictions"]
    indices = values["indices"]
    for subject in sorted(set(dataset.subjects.tolist())):
        mask = dataset.subjects[indices] == subject
        rows.append({
            "subject": int(subject),
            "trials": int(mask.sum()),
            "accuracy": float(accuracy_score(labels[mask], predictions[mask])),
            "f1_macro": float(f1_score(labels[mask], predictions[mask], average="macro", zero_division=0)),
        })
    return rows


def main() -> None:
    args = parse_args()
    config = read_config(args)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)
    seed_everything(
        int(config["seed"]), bool(config["deterministic"]), bool(config["cudnn_benchmark"])
    )
    arrays = load_arrays(config)
    stats = fit_train_fnirs_stats(arrays["train"].fnirs_um)
    datasets = {
        split: SHINTrialDataset(item, stats, "hbo_hbr")
        for split, item in arrays.items()
    }
    loaders = {
        split: make_loader(
            dataset,
            int(config["batch_size"]),
            split == "train",
            int(config["num_workers"]),
            int(config["seed"]),
        )
        for split, dataset in datasets.items()
    }
    diagnostic = run_diagnostic(
        config, datasets, make_loader(datasets["train"], 1, False, 0, int(config["seed"])), output_dir
    )
    if config["diagnose_only"]:
        print(json.dumps(diagnostic, ensure_ascii=False, indent=2), flush=True)
        return
    if str(config["device"]).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(config["device"])
    model, load_record = build_model(config)
    model.to(device)
    optimizer = make_optimizer(model, config)
    best_score = (-1.0, -1.0, -math.inf)
    best_state = None
    best_epoch = 0
    best_val = None
    no_improvement = 0
    history = []
    started = time.time()
    patience = config["early_stopping_patience"]
    for epoch in range(1, int(config["epochs"]) + 1):
        train = train_epoch(model, loaders["train"], optimizer, device, config)
        val, _ = evaluate(model, loaders["val"], device)
        fusion = val["fusion"]
        score = (fusion["accuracy"], fusion["f1_macro"], -fusion["loss"])
        improved = score > best_score
        if improved:
            best_score = score
            best_state = trainable_state(model)
            best_epoch = epoch
            best_val = val
            no_improvement = 0
        else:
            no_improvement += 1
        record = {
            "epoch": epoch,
            "train": train,
            "val": val,
            "is_best": improved,
            "epochs_without_improvement": no_improvement,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        write_json(output_dir / "history.json", history)
        print(
            f"{config['task']} {config['prompt_mode']} epoch={epoch:03d} "
            f"loss={train['total_loss']:.4f} val_eeg={val['eeg']['accuracy']:.4f} "
            f"val_fusion={fusion['accuracy']:.4f} gate={val['gate']:.5f}",
            flush=True,
        )
        if patience is not None and no_improvement >= int(patience):
            break
    if best_state is None:
        raise RuntimeError("No best state recorded")
    load_trainable_state(model, best_state)
    test, test_values = evaluate(model, loaders["test"], device)
    torch.save(
        {"trainable_model": best_state, "best_epoch": best_epoch, "config": config},
        output_dir / "best_trainable.pt",
    )
    summary = {
        "task": config["task"],
        "step": PROMPT_STEPS[config["prompt_mode"]],
        "prompt_mode": config["prompt_mode"],
        "model": (
            "fNIRS-T learned context + sample prompt concatenation + "
            "final-layer universal residual cross-attention"
            if config["prompt_mode"] == "learned_context_sample"
            else "fNIRS-T unified prompt + final-layer universal residual cross-attention"
        ),
        "input_epochs": {
            "eeg_seconds": [
                config["eeg_epoch_start_s"], config["eeg_epoch_stop_s"]
            ],
            "fnirs_seconds": [
                config["fnirs_epoch_start_s"], config["fnirs_epoch_stop_s"]
            ],
            "fnirs_sampling_points": config["fnirs_sampling_points"],
        },
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "stopped_early": len(history) < int(config["epochs"]),
        "best_val": best_val,
        "best_test": test,
        "fusion_gain": test["fusion"]["accuracy"] - test["eeg"]["accuracy"],
        "test_subjects": {
            branch: subject_metrics(datasets["test"], test_values, branch)
            for branch in BRANCHES
        },
        "test_values": serializable_values(test_values),
        "reference": load_record,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
