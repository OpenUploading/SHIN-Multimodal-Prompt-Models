# -*- coding: utf-8 -*-
"""Train frozen CBraMod with fNIRS graph-conditioned gated K/V adapters."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
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

from cbramod_gated_kv_prompt import GatedCBraModGraphPrompt
from shin_multimodal_data import (
    SHINTrialDataset,
    SHIN_EEG_CHANNELS,
    TASKS,
    fit_train_fnirs_stats,
    load_fnirs_montage,
    load_split,
)


BRANCHES = ("eeg", "fnirs", "fusion")
EEG_SCALE_DIVISOR = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SHIN fNIRS graph prompt -> frozen CBraMod gated K/V adapters"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--max-subjects-per-split", type=int)
    parser.add_argument("--device")
    return parser.parse_args()


def read_config(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    required = {
        "task", "fnirs_modalities", "eeg_root", "fnirs_root", "cbramod_root",
        "checkpoint", "cache_dir", "seed", "epochs", "batch_size",
        "graph_lr", "adapter_lr", "head_lr", "weight_decay",
        "loss_weights", "prompt_layer_indices",
        "eeg_scale_divisor", "early_stopping_patience",
        "chromophore_encoder_mode", "prompt_stream_mode",
        "sgformer_attention_layers", "sgformer_heads",
        "sgformer_attention_residual_weight", "sgformer_graph_weight",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Config is missing fields: {missing}")
    config["config_source"] = str(args.config.resolve())
    config["output_dir"] = str(args.output_dir.resolve())
    config["diagnose_only"] = bool(args.diagnose_only)
    config["max_subjects_per_split"] = args.max_subjects_per_split
    if args.device:
        config["device"] = args.device
    config.setdefault("device", "cuda")
    config.setdefault("epoch_start_s", 0.0)
    config.setdefault("epoch_stop_s", 10.0)
    config.setdefault("graph_dimension", 128)
    config.setdefault("dropout", 0.1)
    config.setdefault("num_workers", 0)
    config.setdefault("deterministic", True)
    config.setdefault("cudnn_benchmark", False)
    config.setdefault("mixed_precision", False)
    if config["task"] not in TASKS:
        raise ValueError(f"Unknown task: {config['task']}")
    if config["fnirs_modalities"] not in SHINTrialDataset.MODALITY_INDICES:
        raise ValueError(f"Unknown fNIRS modalities: {config['fnirs_modalities']}")
    if config["chromophore_encoder_mode"] not in {"joint", "separate_concat"}:
        raise ValueError(
            "chromophore_encoder_mode must be 'joint' or 'separate_concat'"
        )
    if (
        config["chromophore_encoder_mode"] == "separate_concat"
        and config["fnirs_modalities"] != "hbo_hbr"
    ):
        raise ValueError(
            "separate_concat requires fnirs_modalities='hbo_hbr' in HbO,HbR order"
        )
    if config["prompt_stream_mode"] not in {"shared", "split_spatial_temporal"}:
        raise ValueError(
            "prompt_stream_mode must be 'shared' or 'split_spatial_temporal'"
        )
    if set(config["loss_weights"]) != set(BRANCHES):
        raise ValueError(f"loss_weights must contain exactly {BRANCHES}")
    if int(config["epochs"]) != 50:
        raise ValueError("This first gated experiment is fixed to 50 epochs")
    if sorted(config["prompt_layer_indices"]) != [8, 9, 10, 11]:
        raise ValueError("First gated experiment must inject only into blocks 8-11")
    if float(config["eeg_scale_divisor"]) != EEG_SCALE_DIVISOR:
        raise ValueError("CBraMod input must use the paper's physical_uV / 100 scaling")
    patience = config["early_stopping_patience"]
    if config["task"] == "ma" and int(patience or 0) != 15:
        raise ValueError("MA must use validation early stopping with patience=15")
    if config["task"] == "mi" and patience is not None:
        raise ValueError("MI keeps the fixed 50-epoch schedule without early stopping")
    if int(config["sgformer_attention_layers"]) != 1:
        raise ValueError("The paper-faithful SGFormer experiment uses one attention layer")
    if int(config["sgformer_heads"]) != 1:
        raise ValueError("The paper-faithful SGFormer experiment uses one attention head")
    for name in ("sgformer_attention_residual_weight", "sgformer_graph_weight"):
        if not 0.0 <= float(config[name]) <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")
    return config


def seed_everything(seed: int, deterministic: bool, benchmark: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def subject_ids(start: int, stop: int, limit: int | None) -> list[int]:
    values = list(range(start, stop + 1))
    return values[:limit] if limit else values


def load_cbramod(config: dict[str, Any]) -> tuple[nn.Module, dict]:
    root = Path(config["cbramod_root"]).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CBraMod root not found: {root}")
    sys.path.insert(0, str(root))
    from models.cbramod import CBraMod

    backbone = CBraMod(
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,
        seq_len=30,
        n_layer=12,
        nhead=8,
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
        "proj_out_after_loading": "Identity",
    }


def make_loader(
    dataset: SHINTrialDataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)

    def worker_init(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

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
    if eeg.shape[1:] != (30, 2000):
        raise ValueError(f"Expected trial EEG [B,30,2000], got {tuple(eeg.shape)}")
    return eeg.reshape(eeg.shape[0], 30, 10, 200)


def metric_dict(labels: np.ndarray, predictions: np.ndarray, loss: float) -> dict:
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "cohen_kappa": float(cohen_kappa_score(labels, predictions)),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }


@torch.no_grad()
def evaluate(
    model: GatedCBraModGraphPrompt,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    labels: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    logits: dict[str, list[np.ndarray]] = {branch: [] for branch in BRANCHES}
    losses = {branch: 0.0 for branch in BRANCHES}
    for eeg, fnirs, target, index in loader:
        target_device = target.to(device, non_blocking=True)
        output = model(
            prepare_eeg(eeg, device),
            fnirs.to(device, non_blocking=True).float(),
        )
        labels.append(target.numpy())
        indices.append(index.numpy())
        for branch in BRANCHES:
            losses[branch] += float(criterion(output[branch], target_device).item())
            logits[branch].append(output[branch].cpu().numpy())
    y_true = np.concatenate(labels)
    item_indices = np.concatenate(indices)
    values: dict[str, np.ndarray] = {
        "labels": y_true,
        "indices": item_indices,
    }
    metrics: dict[str, dict] = {}
    for branch in BRANCHES:
        branch_logits = np.concatenate(logits[branch])
        probabilities = torch.softmax(
            torch.from_numpy(branch_logits), dim=1
        ).numpy()
        predictions = branch_logits.argmax(axis=1)
        metrics[branch] = metric_dict(
            y_true, predictions, losses[branch] / len(y_true)
        )
        values[f"{branch}_probabilities"] = probabilities
        values[f"{branch}_predictions"] = predictions
    return metrics, values


def train_epoch(
    model: GatedCBraModGraphPrompt,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: dict[str, float],
) -> dict[str, float]:
    model.train()
    # Frozen backbone must retain deterministic eval-mode behavior.
    model.backbone.eval()
    criterion = nn.CrossEntropyLoss()
    running = {branch: 0.0 for branch in BRANCHES}
    running["total"] = 0.0
    predictions: dict[str, list[np.ndarray]] = {
        branch: [] for branch in BRANCHES
    }
    targets: list[np.ndarray] = []
    seen = 0
    for eeg, fnirs, target, _ in loader:
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            prepare_eeg(eeg, device),
            fnirs.to(device, non_blocking=True).float(),
        )
        branch_losses = {
            branch: criterion(output[branch], target) for branch in BRANCHES
        }
        total = sum(
            float(loss_weights[branch]) * branch_losses[branch]
            for branch in BRANCHES
        )
        total.backward()
        optimizer.step()
        batch = len(target)
        seen += batch
        targets.append(target.detach().cpu().numpy())
        for branch in BRANCHES:
            running[branch] += float(branch_losses[branch].item()) * batch
            predictions[branch].append(
                output[branch].detach().argmax(dim=1).cpu().numpy()
            )
        running["total"] += float(total.item()) * batch
    result = {f"{key}_loss": value / seen for key, value in running.items()}
    y_true = np.concatenate(targets)
    for branch in BRANCHES:
        result[f"{branch}_accuracy"] = float(
            accuracy_score(y_true, np.concatenate(predictions[branch]))
        )
    return result


def subject_metrics(
    subjects: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict:
    rows = []
    for subject in sorted(set(subjects.tolist())):
        mask = subjects == subject
        rows.append({
            "subject": int(subject),
            "trials": int(mask.sum()),
            "accuracy": float(accuracy_score(labels[mask], predictions[mask])),
            "f1_macro": float(
                f1_score(
                    labels[mask], predictions[mask],
                    average="macro", zero_division=0,
                )
            ),
        })
    accuracy = np.asarray([row["accuracy"] for row in rows])
    macro_f1 = np.asarray([row["f1_macro"] for row in rows])
    return {
        "subjects": rows,
        "accuracy_mean": float(accuracy.mean()),
        "accuracy_std": float(accuracy.std(ddof=0)),
        "f1_macro_mean": float(macro_f1.mean()),
        "f1_macro_std": float(macro_f1.std(ddof=0)),
    }


def write_predictions(
    path: Path,
    dataset: SHINTrialDataset,
    values: dict[str, np.ndarray],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["key", "subject", "true_label"]
        for branch in BRANCHES:
            fields.extend([
                f"{branch}_prediction",
                f"{branch}_probability_0",
                f"{branch}_probability_1",
            ])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for position, item_index in enumerate(values["indices"]):
            index = int(item_index)
            row: dict[str, Any] = {
                "key": str(dataset.keys[index]),
                "subject": int(dataset.subjects[index]),
                "true_label": int(values["labels"][position]),
            }
            for branch in BRANCHES:
                probability = values[f"{branch}_probabilities"][position]
                row[f"{branch}_prediction"] = int(
                    values[f"{branch}_predictions"][position]
                )
                row[f"{branch}_probability_0"] = float(probability[0])
                row[f"{branch}_probability_1"] = float(probability[1])
            writer.writerow(row)


def environment_record(output_dir: Path) -> dict:
    record = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        requirements = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        (output_dir / "requirements.txt").write_text(
            requirements, encoding="utf-8"
        )
    except (subprocess.SubprocessError, OSError) as error:
        record["requirements_error"] = str(error)
    write_json(output_dir / "environment.json", record)
    return record


def report_markdown(config: dict[str, Any], summary: dict) -> str:
    task = TASKS[config["task"]]
    best = summary["best_val"]
    test = summary["best_test"]
    return f"""# SHIN fNIRS SGFormer Gated K/V Prompt -> CBraMod

## 方法

本实现不修改 CBraMod 官方源码。每个 fNIRS 节点的 HbO、HbR 先由参数完全独立的时序卷积编码器分别提取 64维特征，再按 `[HbO,HbR]` 固定顺序拼接为 128维节点表示。时间 Prompt 由该纯时序拼接特征直接投影得到；空间 Prompt 则加入几何与 Node-ID 后进入 SGFormer，再投影得到。两路共享同一个 128→200 投影层，不增加实验参数。CBraMod 最后4个 CrissCross blocks 的空间注意力只使用空间 Prompt K/V，时间注意力只使用时间 Prompt K/V，并通过零初始化标量门控添加到原空间/时间 self-attention 之外：

`Y = SelfAttention(EEG) + tanh(alpha) * CrossAttention(Q_EEG, K_prompt, V_prompt)`

EEG 按原论文在进入 CBraMod 前执行物理微伏除以100；CBraMod 全部参数保持冻结。EEG 与 Fusion 共用官方 all-patch 分类头，避免复制约1.2亿参数；fNIRS 图分支使用单独线性头。

## 固定协议

- 任务：{task["name"]}，{task["description"]}
- 划分：train sub-1~19；val sub-20~24；test sub-25~29
- Seed：{config["seed"]}
- 最大 Epoch：{config["epochs"]}
- 实际完成 Epoch：{summary["epochs_completed"]}
- Early stopping patience：{config["early_stopping_patience"]}
- EEG 缩放：physical uV / {config["eeg_scale_divisor"]}
- Hb 编码：{config["chromophore_encoder_mode"]}（HbO 64维 + HbR 64维 → concat 128维）
- Prompt 流：{config["prompt_stream_mode"]}（时间=CNN concat；空间=geometry+Node-ID+SGFormer）
- Batch：{config["batch_size"]}
- CBraMod：全部冻结
- Prompt blocks：{config["prompt_layer_indices"]}
- SGFormer：1 attention layer，1 head，beta={config["sgformer_attention_residual_weight"]}，graph weight={config["sgformer_graph_weight"]}
- Graph/Adapter/Head LR：{config["graph_lr"]} / {config["adapter_lr"]} / {config["head_lr"]}
- checkpoint 选择：Fusion Val Accuracy
- 主评价单位：trial

## 结果

| 项目 | EEG | fNIRS | Fusion |
|---|---:|---:|---:|
| Best Val Acc | {best["eeg"]["accuracy"]:.4f} | {best["fnirs"]["accuracy"]:.4f} | {best["fusion"]["accuracy"]:.4f} |
| Test Acc | {test["eeg"]["accuracy"]:.4f} | {test["fnirs"]["accuracy"]:.4f} | {test["fusion"]["accuracy"]:.4f} |
| Test Macro-F1 | {test["eeg"]["f1_macro"]:.4f} | {test["fnirs"]["f1_macro"]:.4f} | {test["fusion"]["f1_macro"]:.4f} |
| Test Kappa | {test["eeg"]["cohen_kappa"]:.4f} | {test["fnirs"]["cohen_kappa"]:.4f} | {test["fusion"]["cohen_kappa"]:.4f} |

Best epoch：{summary["best_epoch"]}。
Early stopped：{summary["stopped_early"]}。
Fusion gain：{summary["fusion_gain"]:.4f}。
"""


def main() -> None:
    args = parse_args()
    config = read_config(args)
    output_dir = Path(config["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "mplconfig"))
    seed_everything(
        int(config["seed"]),
        bool(config["deterministic"]),
        bool(config["cudnn_benchmark"]),
    )
    write_json(output_dir / "config.json", config)
    environment_record(output_dir)

    split_subjects = {
        "train": subject_ids(1, 19, config["max_subjects_per_split"]),
        "val": subject_ids(20, 24, config["max_subjects_per_split"]),
        "test": subject_ids(25, 29, config["max_subjects_per_split"]),
    }
    eeg_root = Path(config["eeg_root"]).resolve()
    fnirs_root = Path(config["fnirs_root"]).resolve()
    cache_dir = Path(config["cache_dir"]).resolve()
    arrays = {
        split: load_split(
            eeg_root=eeg_root,
            fnirs_root=fnirs_root,
            subjects=subjects,
            split_name=split,
            task_key=config["task"],
            cache_dir=cache_dir,
            epoch_start_s=float(config["epoch_start_s"]),
            epoch_stop_s=float(config["epoch_stop_s"]),
        )
        for split, subjects in split_subjects.items()
    }
    train_stats = fit_train_fnirs_stats(arrays["train"].fnirs_um)
    np.savez(
        output_dir / "fnirs_train_normalization.npz",
        mean=train_stats["mean"],
        std=train_stats["std"],
    )
    montage = load_fnirs_montage(fnirs_root)
    diagnostics = {
        "protocol": {
            "task": TASKS[config["task"]],
            "splits": split_subjects,
            "seed": config["seed"],
            "selection_metric": "fusion_val_accuracy",
            "evaluation_unit": "trial",
            "backbone_frozen_all_epochs": True,
        },
        "method": {
            "name": "split temporal-CNN/spatial-SGFormer gated K/V prompt adapter",
            "official_implementation": False,
            "cbramod_source_modified": False,
            "prompt_layers_zero_based": config["prompt_layer_indices"],
            "spatial_temporal_gates_initialized_zero": True,
            "shared_eeg_fusion_classifier": True,
            "eeg_scale_divisor": float(config["eeg_scale_divisor"]),
            "early_stopping_patience": config["early_stopping_patience"],
            "chromophore_encoder_mode": config["chromophore_encoder_mode"],
            "chromophore_feature_order": ["HbO", "HbR"],
            "chromophore_branch_dimension": int(config["graph_dimension"]) // 2,
            "chromophore_concat_dimension": int(config["graph_dimension"]),
            "prompt_stream_mode": config["prompt_stream_mode"],
            "temporal_prompt_source": "independent HbO/HbR 1D-CNN concat only",
            "spatial_prompt_source": (
                "temporal concat + geometry + node ID -> SGFormer"
            ),
            "prompt_projection_shared": True,
            "graph_encoder": "SGFormer",
            "sgformer_attention_layers": config["sgformer_attention_layers"],
            "sgformer_heads": config["sgformer_heads"],
            "sgformer_attention_residual_weight": config[
                "sgformer_attention_residual_weight"
            ],
            "sgformer_graph_weight": config["sgformer_graph_weight"],
        },
        "alignment": {
            "eeg_sampling_rate_hz": 200,
            "fnirs_sampling_rate_hz": 10,
            "epoch_seconds": [config["epoch_start_s"], config["epoch_stop_s"]],
            "eeg_channels": SHIN_EEG_CHANNELS,
            "fnirs_chromophore_order": ["HbO", "HbR"],
            "unique_key": "subject/task/session/trial/start",
        },
        "preprocessing": {
            "eeg": (
                "physical_uV -> common-average reference -> causal fifth-order "
                "Butterworth 0.3-50 Hz -> physical_uV / 100 -> "
                "reshape [30,10,200]"
            ),
            "fnirs": (
                "intensity -> optical density -> modified Beer-Lambert -> "
                "HbO/HbR micromolar -> 0.01-0.1 Hz -> baseline -5..-2 s"
            ),
            "normalization": (
                "node/chromophore mean and std fitted on train subjects only"
            ),
            "temporal_encoding": (
                "independent HbO and HbR Conv1d encoders; 64-D per branch; "
                "concatenate [HbO,HbR] to 128-D before SGFormer"
            ),
        },
        "graph": {
            "nodes": 36,
            "edge_count": int(montage["edge_index"].shape[1]),
            "method": montage["graph_method"],
            "prompt_tokens": 36,
            "prompt_dimension": 200,
            "temporal_prompt_tokens": 36,
            "spatial_prompt_tokens": 36,
            "encoder": "one-layer one-head SGFormer global attention + shallow local GCN",
        },
        "splits": {
            split: {
                "subjects": split_subjects[split],
                "eeg_shape": list(item.eeg_uv.shape),
                "fnirs_shape": list(item.fnirs_um.shape),
                "label_counts": {
                    str(key): int(value)
                    for key, value in Counter(item.labels.tolist()).items()
                },
                "nan_count": int(
                    np.isnan(item.eeg_uv).sum() + np.isnan(item.fnirs_um).sum()
                ),
                "inf_count": int(
                    np.isinf(item.eeg_uv).sum() + np.isinf(item.fnirs_um).sum()
                ),
                "unique_keys": len(set(item.keys.tolist())),
                "trials": len(item.labels),
                "details": item.details,
            }
            for split, item in arrays.items()
        },
    }
    write_json(output_dir / "diagnostics.json", diagnostics)

    datasets = {
        split: SHINTrialDataset(
            item, train_stats, config["fnirs_modalities"]
        )
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

    backbone, load_record = load_cbramod(config)
    model = GatedCBraModGraphPrompt(
        backbone=backbone,
        positions_3d=torch.from_numpy(montage["positions_3d"]),
        edge_index=torch.from_numpy(montage["edge_index"]),
        chromophores=len(
            SHINTrialDataset.MODALITY_INDICES[config["fnirs_modalities"]]
        ),
        chromophore_encoder_mode=config["chromophore_encoder_mode"],
        prompt_stream_mode=config["prompt_stream_mode"],
        prompt_layer_indices=list(config["prompt_layer_indices"]),
        graph_dimension=int(config["graph_dimension"]),
        dropout=float(config["dropout"]),
        sgformer_attention_residual_weight=float(
            config["sgformer_attention_residual_weight"]
        ),
        sgformer_graph_weight=float(config["sgformer_graph_weight"]),
    )
    model.freeze_backbone()
    diagnostics["pretrained_load"] = load_record
    diagnostics["parameters"] = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "frozen_backbone": sum(
            parameter.numel() for parameter in model.backbone.parameters()
        ),
        "graph_prompt": sum(
            parameter.numel() for parameter in model.graph_prompt.parameters()
        ),
        "sgformer": sum(
            parameter.numel()
            for parameter in model.graph_prompt.sgformer.parameters()
        ),
        "hbo_temporal_encoder": sum(
            parameter.numel()
            for parameter in model.graph_prompt.hbo_temporal_encoder.parameters()
        ) if model.graph_prompt.hbo_temporal_encoder is not None else None,
        "hbr_temporal_encoder": sum(
            parameter.numel()
            for parameter in model.graph_prompt.hbr_temporal_encoder.parameters()
        ) if model.graph_prompt.hbr_temporal_encoder is not None else None,
        "kv_adapters": sum(
            parameter.numel() for parameter in model.adapters.parameters()
        ),
        "shared_classifier": sum(
            parameter.numel() for parameter in model.classifier.parameters()
        ),
        "fnirs_head": sum(
            parameter.numel() for parameter in model.fnirs_head.parameters()
        ),
    }
    diagnostics["initial_gates"] = model.gate_values()
    write_json(output_dir / "diagnostics.json", diagnostics)

    smoke_device = torch.device("cpu")
    model.to(smoke_device)
    eeg, fnirs, smoke_target, _ = next(iter(loaders["train"]))
    model.eval()
    if config["chromophore_encoder_mode"] == "separate_concat":
        hbo_parameters = {
            id(parameter)
            for parameter in model.graph_prompt.hbo_temporal_encoder.parameters()
        }
        hbr_parameters = {
            id(parameter)
            for parameter in model.graph_prompt.hbr_temporal_encoder.parameters()
        }
        with torch.no_grad():
            smoke_batch, smoke_nodes, _, smoke_samples = fnirs[:1].shape
            hbo_smoke = model.graph_prompt.hbo_temporal_encoder(
                fnirs[:1, :, 0, :].reshape(
                    smoke_batch * smoke_nodes, 1, smoke_samples
                ).float()
            ).squeeze(-1).reshape(smoke_batch, smoke_nodes, -1)
            hbr_smoke = model.graph_prompt.hbr_temporal_encoder(
                fnirs[:1, :, 1, :].reshape(
                    smoke_batch * smoke_nodes, 1, smoke_samples
                ).float()
            ).squeeze(-1).reshape(smoke_batch, smoke_nodes, -1)
            concat_smoke = torch.cat([hbo_smoke, hbr_smoke], dim=-1)
        diagnostics["chromophore_encoder_smoke"] = {
            "input_order": ["HbO", "HbR"],
            "hbo_shape": list(hbo_smoke.shape),
            "hbr_shape": list(hbr_smoke.shape),
            "concatenated_shape": list(concat_smoke.shape),
            "shared_parameter_count": len(hbo_parameters & hbr_parameters),
            "independent_parameters_passed": not (hbo_parameters & hbr_parameters),
            "finite_outputs_passed": bool(
                torch.isfinite(hbo_smoke).all()
                and torch.isfinite(hbr_smoke).all()
                and torch.isfinite(concat_smoke).all()
            ),
        }
    prepared_smoke_eeg = prepare_eeg(eeg[:1], smoke_device)
    diagnostics["eeg_input_scaling"] = {
        "source_unit": "physical_uV",
        "divisor": EEG_SCALE_DIVISOR,
        "model_input_unit": "100_uV",
        "raw_max_abs_uV": float(eeg[:1].abs().max().item()),
        "prepared_max_abs": float(prepared_smoke_eeg.abs().max().item()),
        "division_ratio": float(
            eeg[:1].abs().max().item()
            / prepared_smoke_eeg.abs().max().item()
        ),
        "passed": bool(torch.allclose(
            prepared_smoke_eeg.reshape_as(eeg[:1]) * EEG_SCALE_DIVISOR,
            eeg[:1].float(),
            rtol=1e-6,
            atol=1e-6,
        )),
    }
    with torch.no_grad():
        smoke = model(
            prepared_smoke_eeg,
            fnirs[:1].float(),
        )
    diagnostics["model_smoke"] = {
        key: list(value.shape) for key, value in smoke.items()
    }
    prompt_difference = (
        smoke["spatial_prompt"] - smoke["temporal_prompt"]
    ).abs()
    diagnostics["prompt_stream_smoke"] = {
        "mode": config["prompt_stream_mode"],
        "spatial_prompt_shape": list(smoke["spatial_prompt"].shape),
        "temporal_prompt_shape": list(smoke["temporal_prompt"].shape),
        "mean_abs_difference": float(prompt_difference.mean().item()),
        "max_abs_difference": float(prompt_difference.max().item()),
        "distinct_content_passed": bool(prompt_difference.max().item() > 0.0),
        "finite_outputs_passed": bool(
            torch.isfinite(smoke["spatial_prompt"]).all()
            and torch.isfinite(smoke["temporal_prompt"]).all()
        ),
        "shared_projection_parameter_count": sum(
            parameter.numel()
            for parameter in model.graph_prompt.projection.parameters()
        ),
    }
    diagnostics["zero_gate_equivalence"] = {
        "max_abs_eeg_fusion_logit_difference": float(
            (smoke["eeg"] - smoke["fusion"]).abs().max().item()
        ),
        "passed": bool(torch.equal(smoke["eeg"], smoke["fusion"])),
    }

    # Use a small nonzero diagnostic gate only for one backward pass, proving
    # that every adapter path can receive gradients. Restore exact zero after.
    with torch.no_grad():
        for adapter in model.adapters.values():
            adapter.spatial_gate.fill_(0.01)
            adapter.temporal_gate.fill_(0.01)
    model.train()
    model.backbone.eval()
    smoke_output = model(
        prepared_smoke_eeg,
        fnirs[:1].float(),
    )
    smoke_loss = sum(
        nn.functional.cross_entropy(
            smoke_output[branch], smoke_target[:1].to(smoke_device)
        )
        for branch in BRANCHES
    )
    smoke_loss.backward()
    trainable = {
        name: parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    missing_gradients = [
        name for name, parameter in trainable.items()
        if parameter.grad is None
    ]
    nonfinite_gradients = [
        name for name, parameter in trainable.items()
        if parameter.grad is not None
        and not torch.isfinite(parameter.grad).all()
    ]
    diagnostics["gradient_smoke_at_gate_0p01"] = {
        "loss": float(smoke_loss.item()),
        "trainable_tensors": len(trainable),
        "missing_gradients": missing_gradients,
        "nonfinite_gradients": nonfinite_gradients,
        "passed": not missing_gradients and not nonfinite_gradients,
    }
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for adapter in model.adapters.values():
            adapter.spatial_gate.zero_()
            adapter.temporal_gate.zero_()
    diagnostics["restored_gates"] = model.gate_values()
    write_json(output_dir / "diagnostics.json", diagnostics)
    if config["diagnose_only"]:
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2), flush=True)
        return

    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(config["device"])
    model.to(device)
    optimizer = torch.optim.AdamW([
        {
            "params": list(model.graph_prompt.parameters()),
            "lr": float(config["graph_lr"]),
            "name": "graph_prompt",
        },
        {
            "params": list(model.adapters.parameters()),
            "lr": float(config["adapter_lr"]),
            "name": "kv_adapters",
        },
        {
            "params": list(model.classifier.parameters())
            + list(model.fnirs_head.parameters()),
            "lr": float(config["head_lr"]),
            "name": "heads",
        },
    ], weight_decay=float(config["weight_decay"]))

    history: list[dict] = []
    best_accuracy = -1.0
    best_epoch = 0
    best_val: dict[str, dict] | None = None
    patience = config["early_stopping_patience"]
    epochs_without_improvement = 0
    stopped_early = False
    stop_epoch: int | None = None
    started = time.time()
    for epoch in range(1, int(config["epochs"]) + 1):
        train_metrics = train_epoch(
            model, loaders["train"], optimizer, device, config["loss_weights"]
        )
        val_metrics, _ = evaluate(model, loaders["val"], device)
        record = {
            "epoch": epoch,
            "stage": "frozen_cbramod_split_temporal_cnn_spatial_sgformer_gated_kv_prompt",
            "train": train_metrics,
            "val": val_metrics,
            "gate_values": model.gate_values(),
            "learning_rates": {
                group["name"]: float(group["lr"])
                for group in optimizer.param_groups
            },
            "elapsed_seconds": time.time() - started,
            "is_best": False,
        }
        if val_metrics["fusion"]["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["fusion"]["accuracy"]
            best_epoch = epoch
            best_val = val_metrics
            record["is_best"] = True
            epochs_without_improvement = 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "config": config,
            }, output_dir / "best.pt")
        else:
            epochs_without_improvement += 1
        record["epochs_without_improvement"] = epochs_without_improvement
        record["early_stopping_patience"] = patience
        history.append(record)
        write_json(output_dir / "history.json", history)
        gates = model.gate_values()
        print(
            f"epoch {epoch:03d}/{config['epochs']} "
            f"loss={train_metrics['total_loss']:.4f} "
            f"val(eeg/fnirs/fusion)="
            f"{val_metrics['eeg']['accuracy']:.4f}/"
            f"{val_metrics['fnirs']['accuracy']:.4f}/"
            f"{val_metrics['fusion']['accuracy']:.4f} "
            f"gates(last)=s:{gates['11']['spatial']:.4f},"
            f"t:{gates['11']['temporal']:.4f}",
            flush=True,
        )
        if patience is not None and epochs_without_improvement >= int(patience):
            stopped_early = True
            stop_epoch = epoch
            print(
                f"early stopping at epoch {epoch}: no Fusion Val Accuracy "
                f"improvement for {patience} epochs (best epoch={best_epoch})",
                flush=True,
            )
            break

    final_gate_values = model.gate_values()
    final_test, final_values = evaluate(model, loaders["test"], device)
    checkpoint = torch.load(
        output_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    best_test, best_values = evaluate(model, loaders["test"], device)
    write_predictions(
        output_dir / "trial_predictions_best.csv",
        datasets["test"],
        best_values,
    )
    write_predictions(
        output_dir / "trial_predictions_final.csv",
        datasets["test"],
        final_values,
    )
    per_subject = {
        branch: subject_metrics(
            datasets["test"].subjects[best_values["indices"]],
            best_values["labels"],
            best_values[f"{branch}_predictions"],
        )
        for branch in BRANCHES
    }
    fusion_gain = (
        best_test["fusion"]["accuracy"]
        - max(best_test["eeg"]["accuracy"], best_test["fnirs"]["accuracy"])
    )
    summary = {
        "experiment_id": output_dir.name,
        "model": "CBraMod + split temporal-CNN/spatial-SGFormer gated K/V prompts",
        "task": config["task"],
        "task_name": TASKS[config["task"]]["name"],
        "modalities": ["eeg", "hbo", "hbr"],
        "seed": config["seed"],
        "splits": split_subjects,
        "backbone_frozen_all_epochs": True,
        "prompt_layer_indices": config["prompt_layer_indices"],
        "graph_encoder": "SGFormer",
        "chromophore_encoder_mode": config["chromophore_encoder_mode"],
        "prompt_stream_mode": config["prompt_stream_mode"],
        "temporal_prompt_source": "independent HbO/HbR 1D-CNN concat only",
        "spatial_prompt_source": (
            "temporal concat + geometry + node ID -> SGFormer"
        ),
        "chromophore_feature_order": ["HbO", "HbR"],
        "chromophore_branch_dimension": int(config["graph_dimension"]) // 2,
        "sgformer_attention_layers": config["sgformer_attention_layers"],
        "sgformer_heads": config["sgformer_heads"],
        "sgformer_attention_residual_weight": config[
            "sgformer_attention_residual_weight"
        ],
        "sgformer_graph_weight": config["sgformer_graph_weight"],
        "eeg_scale_divisor": float(config["eeg_scale_divisor"]),
        "early_stopping_patience": patience,
        "stopped_early": stopped_early,
        "stop_epoch": stop_epoch,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val": best_val,
        "best_test": best_test,
        "final_test": final_test,
        "branch_results": best_test,
        "fusion_gain": float(fusion_gain),
        "best_gate_values": model.gate_values(),
        "final_gate_values": final_gate_values,
        "per_subject": per_subject,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "per_subject_metrics.json", per_subject)
    (output_dir / "EXPERIMENT_RECORD.md").write_text(
        report_markdown(config, summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
