# -*- coding: utf-8 -*-
"""Train frozen CBraMod with an explicit NVE-SGFormer fNIRS prompt."""

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
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader

from cbramod_gated_kv_prompt import GatedCBraModGraphPrompt

PROJECT_DIR = Path(__file__).resolve().parent

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
        description="SHIN NVE-SGFormer prompt -> frozen CBraMod gated K/V adapters"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--max-subjects-per-split", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
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
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.smoke:
        config["smoke"] = True
        config["epochs"] = 2
        config["stage1_epochs"] = 2
    config.setdefault("device", "cuda")
    config.setdefault("epoch_start_s", 0.0)
    config.setdefault("epoch_stop_s", 10.0)
    config.setdefault("graph_dimension", 128)
    config.setdefault("dropout", 0.1)
    config.setdefault("num_workers", 0)
    config.setdefault("deterministic", True)
    config.setdefault("cudnn_benchmark", False)
    config.setdefault("mixed_precision", False)
    config.setdefault("prompt_generator", "original")
    config.setdefault("mope_expert_count", 4)
    config.setdefault("mope_temperature", 0.1)
    config.setdefault("mope_router_noise_std", 0.00390625)
    config.setdefault("mope_importance_threshold", 0.05)
    config.setdefault("mope_condition_dim", 128)
    config.setdefault("mope_top_k", None)
    config.setdefault("prompt_components", "all")
    config.setdefault("prompt_branch_mode", "both")
    config.setdefault("nve_spatial_encoder", "sgformer")
    config.setdefault("temporal_prompt_mode", "node_summary")
    config.setdefault("temporal_kv_policy", "all")
    config.setdefault("temporal_future_steps", 3)
    config.setdefault("importance_lambda", 0.0)
    config.setdefault("training_protocol", "joint")
    config.setdefault("fnirs_control", "aligned")
    config.setdefault("stage1_epochs", 50)
    config.setdefault("stage1_checkpoint_source", None)
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
    if int(config["epochs"]) != 50 and not bool(config.get("smoke")):
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
    expected_sgformer_layers = (
        2
        if config["nve_spatial_encoder"] == "two_layer_sgformer_bottleneck"
        else 1
    )
    if int(config["sgformer_attention_layers"]) != expected_sgformer_layers:
        raise ValueError(
            f"{config['nve_spatial_encoder']} requires "
            f"sgformer_attention_layers={expected_sgformer_layers}"
        )
    if int(config["sgformer_heads"]) != 1:
        raise ValueError("The paper-faithful SGFormer experiment uses one attention head")
    for name in ("sgformer_attention_residual_weight", "sgformer_graph_weight"):
        if not 0.0 <= float(config[name]) <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")
    if config["prompt_generator"] not in {
        "original", "mope", "nve_sgformer", "cnn_sgformer"
    }:
        raise ValueError(
            "prompt_generator must be 'original', 'mope', 'nve_sgformer', or "
            "'cnn_sgformer', got "
            f"{config['prompt_generator']!r}"
        )
    if config["prompt_generator"] == "nve_sgformer":
        if config["chromophore_encoder_mode"] != "separate_concat":
            raise ValueError("NVE-SGFormer keeps independent HbO/HbR temporal encoders")
        if config["nve_spatial_encoder"] not in {
            "sgformer", "two_layer_sgformer_bottleneck", "hybrid_sgformer",
            "identity", "two_layer_gcn"
        }:
            raise ValueError(
                "nve_spatial_encoder must be 'sgformer', "
                "'two_layer_sgformer_bottleneck', 'hybrid_sgformer', "
                "'identity', or 'two_layer_gcn'"
            )
        graph_weight = float(config["sgformer_graph_weight"])
        if config["nve_spatial_encoder"] == "hybrid_sgformer":
            if graph_weight <= 0.0:
                raise ValueError(
                    "Hybrid NVE-SGFormer requires sgformer_graph_weight > 0"
                )
        elif graph_weight != 0.0:
            raise ValueError(
                "Only hybrid_sgformer may use a non-zero SGFormer graph weight"
            )
    elif config["prompt_generator"] == "cnn_sgformer":
        if config["chromophore_encoder_mode"] != "separate_concat":
            raise ValueError("CNN-SGFormer control requires independent HbO/HbR")
        if config["prompt_stream_mode"] != "split_spatial_temporal":
            raise ValueError("CNN-SGFormer control keeps split spatial/temporal prompts")
        if float(config["sgformer_graph_weight"]) != 0.0:
            raise ValueError("CNN-SGFormer matched control is Global-only")
    elif config["nve_spatial_encoder"] != "sgformer":
        raise ValueError("nve_spatial_encoder only applies to NVE experiments")
    if config["prompt_branch_mode"] not in {"both", "spatial_only", "temporal_only"}:
        raise ValueError(
            "prompt_branch_mode must be 'both', 'spatial_only', or 'temporal_only'"
        )
    if config["temporal_prompt_mode"] not in {
        "node_summary", "aligned_10_tokens", "overlap_3s_10_tokens",
        "overlap_3s_region_time_tokens", "overlap_3s_node_time_tokens",
    }:
        raise ValueError(
            "temporal_prompt_mode must be 'node_summary', 'aligned_10_tokens', "
            "'overlap_3s_10_tokens', 'overlap_3s_region_time_tokens', or "
            "'overlap_3s_node_time_tokens'"
        )
    if config["temporal_prompt_mode"] in {
        "aligned_10_tokens", "overlap_3s_10_tokens",
        "overlap_3s_region_time_tokens", "overlap_3s_node_time_tokens",
    }:
        if config["prompt_generator"] != "nve_sgformer":
            raise ValueError("Temporal token modes are currently defined for NVE")
        if config["prompt_stream_mode"] != "split_spatial_temporal":
            raise ValueError("Temporal token modes require split prompt streams")
        if config["prompt_branch_mode"] != "both":
            raise ValueError("Temporal token validation uses both branches")
    if config["temporal_kv_policy"] not in {"all", "current_and_future"}:
        raise ValueError(
            "temporal_kv_policy must be 'all' or 'current_and_future'"
        )
    if not 0 <= int(config["temporal_future_steps"]) <= 9:
        raise ValueError("temporal_future_steps must be in [0,9]")
    if config["temporal_kv_policy"] == "current_and_future":
        if config["temporal_prompt_mode"] != "overlap_3s_node_time_tokens":
            raise ValueError(
                "current_and_future K/V is defined for 36x10 node-time tokens"
            )
        if int(config["temporal_future_steps"]) != 3:
            raise ValueError(
                "The first constrained experiment is fixed to current + 3 future steps"
            )
    if (
        config["prompt_branch_mode"] == "spatial_only"
        and config["prompt_generator"] != "nve_sgformer"
    ):
        raise ValueError("Strict spatial-only mode is currently defined for NVE-SGFormer")
    if config["stage1_checkpoint_source"] is not None:
        source = Path(config["stage1_checkpoint_source"])
        if not source.is_file():
            raise FileNotFoundError(f"Stage-1 checkpoint not found: {source}")
    if config["prompt_components"] not in {"all", "static_only"}:
        raise ValueError(
            "prompt_components must be 'all' or 'static_only', got "
            f"{config['prompt_components']!r}"
        )
    if config["training_protocol"] not in {"joint", "prompt_only"}:
        raise ValueError(
            "training_protocol must be 'joint' or 'prompt_only', got "
            f"{config['training_protocol']!r}"
        )
    if config["fnirs_control"] not in {"aligned", "shuffled"}:
        raise ValueError(
            "fnirs_control must be 'aligned' or 'shuffled', got "
            f"{config['fnirs_control']!r}"
        )
    if int(config["mope_expert_count"]) < 2:
        raise ValueError("mope_expert_count must be >= 2")
    if config["mope_top_k"] is not None and not (
        1 <= int(config["mope_top_k"]) <= int(config["mope_expert_count"])
    ):
        raise ValueError("mope_top_k must be between 1 and mope_expert_count")
    if float(config["mope_temperature"]) <= 0:
        raise ValueError("mope_temperature must be > 0")
    if float(config["mope_router_noise_std"]) < 0:
        raise ValueError("mope_router_noise_std must be >= 0")
    if float(config["mope_importance_threshold"]) < 0:
        raise ValueError("mope_importance_threshold must be >= 0")
    if float(config["importance_lambda"]) < 0:
        raise ValueError("importance_lambda must be >= 0")
    if int(config["stage1_epochs"]) < 1:
        raise ValueError("stage1_epochs must be >= 1")
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
    importance_lambda: float = 0.0,
) -> dict[str, float]:
    model.train()
    # Frozen backbone must retain deterministic eval-mode behavior.
    model.backbone.eval()
    criterion = nn.CrossEntropyLoss()
    running = {branch: 0.0 for branch in BRANCHES}
    running["total"] = 0.0
    running["importance"] = 0.0
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
        importance = model.prompt_importance_loss()
        if importance_lambda > 0 and float(importance) > 0:
            total = total + float(importance_lambda) * importance
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
        running["importance"] += float(importance.item()) * batch
    result = {f"{key}_loss": value / seen for key, value in running.items()}
    result["importance_loss"] = running["importance"] / seen
    result["routing_statistics"] = model.routing_statistics()
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


def report_markdown(config: dict[str, Any], summary: dict) -> str:
    """Write an UTF-8 experiment record for NVE and its matched control."""
    task = TASKS[config["task"]]
    best = summary["best_val"]
    test = summary["best_test"]
    spatial_encoder = {
        "sgformer": "一层 SGFormer 全局线性注意力",
        "hybrid_sgformer": "SGFormer全局线性注意力 + 局部GCN",
        "identity": "Identity（各节点独立，不进行跨节点交互）",
        "two_layer_gcn": "两层归一化GCN（固定97边，局部两跳传播）",
    }.get(
        config["nve_spatial_encoder"],
        "two SGFormer global-attention layers + residual 128-64-128 bottleneck",
    )
    temporal_stream = (
        "严格关闭：不执行 HbO/HbR 时间CNN，也不计算 Temporal K/V"
        if config["prompt_branch_mode"] == "spatial_only"
        else "与空间路径共享同一组36个NVE-SGFormer节点Prompt"
        if config["prompt_stream_mode"] == "shared"
        else "HbO/HbR独立CNN保留10个一秒时间Token，跨节点均值并加入正弦位置编码"
        if config["temporal_prompt_mode"] == "aligned_10_tokens"
        else "由参数独立的 HbO/HbR 1D-CNN 编码并拼接"
    )
    if config["temporal_prompt_mode"] == "overlap_3s_10_tokens":
        temporal_stream = (
            "HbO/HbR independent CNN encoders: ten overlapping 3-second "
            "windows, 1-second stride, zero-padded boundaries, node mean, "
            "and sinusoidal position encoding"
        )
    if config["temporal_prompt_mode"] == "overlap_3s_region_time_tokens":
        temporal_stream = (
            "HbO/HbR independent CNN encoders: six fixed anatomical regions "
            "x ten overlapping 3-second windows, 1-second stride, zero-padded "
            "boundaries, regional mean, time and region position encodings"
        )
    if config["temporal_prompt_mode"] == "overlap_3s_node_time_tokens":
        temporal_stream = (
            "HbO/HbR independent CNN encoders: 36 nodes x ten overlapping "
            "3-second windows, 1-second stride, zero-padded boundaries, "
            "with learned node IDs and sinusoidal time positions"
        )
    if config["prompt_generator"] == "cnn_sgformer":
        title = "SHIN No-NVE Independent-CNN Matched Control -> CBraMod"
        feature_description = (
            "每个fNIRS节点的HbO、HbR分别经过参数独立的1D-CNN形成64维特征，"
            "按固定顺序拼接为128维节点表示；空间路径不计算23维NVE属性。"
        )
        spatial_encoder = "一层 SGFormer 全局线性注意力"
        trainable_label = "CNN/Adapter/Head"
    else:
        title = "SHIN NVE Spatial Encoder Ablation -> CBraMod"
        feature_description = (
            "每个fNIRS节点先形成23维显式神经血管事件属性，包括HbO/HbR的"
            "四段均值、整体幅值与趋势统计，以及HbO/HbR Pearson相关性；"
            "属性经MLP映射并加入三维位置和Node-ID。"
        )
        trainable_label = "NVE/Adapter/Head"
    graph_note = (
        "使用97条固定边"
        if config["nve_spatial_encoder"] in {
            "hybrid_sgformer", "two_layer_gcn"
        }
        else "不使用邻接矩阵或GCN"
    )
    return f"""# {title}

## 方法

{feature_description}本轮空间编码器为：{spatial_encoder}；{graph_note}。
时间路径为：{temporal_stream}。

空间和时间 Prompt 分别注入冻结 CBraMod 第 8–11 层 CrissCross Attention
的并行 K/V 适配器：

`Y = SelfAttention(EEG) + tanh(alpha) * CrossAttention(Q_EEG, K_prompt, V_prompt)`

## 固定协议

- 任务：{task["name"]}，{task["description"]}
- 划分：train sub-1~19，val sub-20~24，test sub-25~29
- Seed：{config["seed"]}
- 最大/实际完成 Epoch：{config["epochs"]} / {summary["epochs_completed"]}
- Early stopping patience：{config["early_stopping_patience"]}
- EEG 缩放：physical uV / {config["eeg_scale_divisor"]}
- Batch：{config["batch_size"]}
- CBraMod：全程冻结；复用固定 EEG 分类头
- Prompt blocks：{config["prompt_layer_indices"]}
- 空间特征来源：{config["prompt_generator"]}
- 空间编码器：{spatial_encoder}
- SGFormer配置（仅sgformer模式生效）：1 layer，1 head，beta={config["sgformer_attention_residual_weight"]}，无 GCN
- {trainable_label} LR：{config["graph_lr"]} / {config["adapter_lr"]} / {config["head_lr"]}
- checkpoint 选择：Fusion validation accuracy
- 主评价单位：trial

## 结果

| 项目 | EEG | fNIRS | Fusion |
|---|---:|---:|---:|
| Best Val Acc | {best["eeg"]["accuracy"]:.4f} | {best["fnirs"]["accuracy"]:.4f} | {best["fusion"]["accuracy"]:.4f} |
| Test Acc | {test["eeg"]["accuracy"]:.4f} | {test["fnirs"]["accuracy"]:.4f} | {test["fusion"]["accuracy"]:.4f} |
| Test Macro-F1 | {test["eeg"]["f1_macro"]:.4f} | {test["fnirs"]["f1_macro"]:.4f} | {test["fusion"]["f1_macro"]:.4f} |
| Test Kappa | {test["eeg"]["cohen_kappa"]:.4f} | {test["fnirs"]["cohen_kappa"]:.4f} | {test["fusion"]["cohen_kappa"]:.4f} |

Best epoch：{summary["best_epoch"]}。Early stopped：{summary["stopped_early"]}。
Fusion gain：{summary["fusion_gain"]:.4f}。
"""


def set_trainable_stage(model: GatedCBraModGraphPrompt, stage: str) -> None:
    """Configure trainable parameters for joint / head / prompt_only stages."""
    if stage not in {"joint", "head", "prompt_only"}:
        raise ValueError(f"Unknown training stage: {stage!r}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    if stage in {"joint", "head"}:
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    if stage == "head":
        # Stage 1 trains only the shared all-patch classifier from EEG.
        return
    if stage in {"joint", "prompt_only"}:
        for module in (model.graph_prompt, model.adapters):
            for parameter in module.parameters():
                parameter.requires_grad = True
        for parameter in model.fnirs_head.parameters():
            parameter.requires_grad = True
        model.enforce_disabled_branch_freezing()


def make_optimizer(
    model: GatedCBraModGraphPrompt,
    config: dict[str, Any],
    stage: str,
) -> torch.optim.Optimizer:
    """AdamW groups matching the training stage."""
    def trainable(parameters):
        return [parameter for parameter in parameters if parameter.requires_grad]

    if stage == "head":
        groups = [{
            "params": trainable(model.classifier.parameters()),
            "lr": float(config["head_lr"]),
            "name": "head",
        }]
    elif stage == "prompt_only":
        groups = [
            {
                "params": trainable(model.graph_prompt.parameters()),
                "lr": float(config["graph_lr"]),
                "name": "graph_prompt",
            },
            {
                "params": trainable(model.adapters.parameters()),
                "lr": float(config["adapter_lr"]),
                "name": "kv_adapters",
            },
            {
                "params": trainable(model.fnirs_head.parameters()),
                "lr": float(config["head_lr"]),
                "name": "fnirs_head",
            },
        ]
    else:
        groups = [
            {
                "params": trainable(model.graph_prompt.parameters()),
                "lr": float(config["graph_lr"]),
                "name": "graph_prompt",
            },
            {
                "params": trainable(model.adapters.parameters()),
                "lr": float(config["adapter_lr"]),
                "name": "kv_adapters",
            },
            {
                "params": trainable(model.classifier.parameters())
                + trainable(model.fnirs_head.parameters()),
                "lr": float(config["head_lr"]),
                "name": "heads",
            },
        ]
    return torch.optim.AdamW(
        groups, weight_decay=float(config["weight_decay"])
    )


def run_stage(
    model: GatedCBraModGraphPrompt,
    loaders: dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    output_dir: Path,
    stage_label: str,
    ckpt_name: str,
    epochs: int,
    loss_weights: dict[str, float],
    patience: int | None,
    started: float,
) -> dict[str, Any]:
    """Train one stage; saves the best fusion-val checkpoint."""
    history: list[dict] = []
    best_accuracy = -1.0
    best_epoch = 0
    best_val: dict[str, dict] | None = None
    epochs_without_improvement = 0
    stopped_early = False
    stop_epoch: int | None = None
    device = next(iter(model.parameters())).device
    for epoch in range(1, int(epochs) + 1):
        train_metrics = train_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            loss_weights,
            float(config["importance_lambda"]),
        )
        val_metrics, _ = evaluate(model, loaders["val"], device)
        record = {
            "epoch": epoch,
            "stage": stage_label,
            "train": train_metrics,
            "val": val_metrics,
            "gate_values": model.gate_values(),
            "prompt_component_gates": model.prompt_component_gates(),
            "routing_statistics": train_metrics.get(
                "routing_statistics", {}
            ),
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
                "stage": stage_label,
            }, output_dir / ckpt_name)
        else:
            epochs_without_improvement += 1
        record["epochs_without_improvement"] = epochs_without_improvement
        record["early_stopping_patience"] = patience
        history.append(record)
        write_json(
            output_dir / f"history_{stage_label}.json", history
        )
        gates = model.gate_values()
        component_gates = model.prompt_component_gates()
        component_text = ",".join(
            f"{key}={value:.4f}"
            for key, value in component_gates.items()
        )
        print(
            f"[{stage_label}] epoch {epoch:03d}/{epochs} "
            f"loss={train_metrics['total_loss']:.4f} "
            f"imp={train_metrics.get('importance_loss', 0.0):.4f} "
            f"val(eeg/fnirs/fusion)="
            f"{val_metrics['eeg']['accuracy']:.4f}/"
            f"{val_metrics['fnirs']['accuracy']:.4f}/"
            f"{val_metrics['fusion']['accuracy']:.4f} "
            f"gates(last)=s:{gates['11']['spatial']:.4f},"
            f"t:{gates['11']['temporal']:.4f} "
            f"components={component_text}",
            flush=True,
        )
        if patience is not None and epochs_without_improvement >= int(patience):
            stopped_early = True
            stop_epoch = epoch
            print(
                f"[{stage_label}] early stopping at epoch {epoch}: "
                f"no Fusion Val Accuracy improvement for {patience} epochs "
                f"(best epoch={best_epoch})",
                flush=True,
            )
            break
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "best_accuracy": best_accuracy,
        "stopped_early": stopped_early,
        "stop_epoch": stop_epoch,
    }


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
    if config["fnirs_control"] == "shuffled":
        rng = np.random.default_rng(int(config["seed"]))
        arrays = {
            split: replace(
                item,
                fnirs_um=item.fnirs_um[rng.permutation(len(item.labels))],
            )
            for split, item in arrays.items()
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
            "prompt_generator": config["prompt_generator"],
            "prompt_components": config["prompt_components"],
            "prompt_branch_mode": config["prompt_branch_mode"],
            "nve_spatial_encoder": config["nve_spatial_encoder"],
            "temporal_prompt_mode": config["temporal_prompt_mode"],
            "temporal_kv_policy": config["temporal_kv_policy"],
            "temporal_future_steps": int(config["temporal_future_steps"]),
            "training_protocol": config["training_protocol"],
            "fnirs_control": config["fnirs_control"],
            "importance_lambda": float(config["importance_lambda"]),
            "mope_expert_count": int(config["mope_expert_count"]),
            "mope_temperature": float(config["mope_temperature"]),
            "mope_router_noise_std": float(config["mope_router_noise_std"]),
            "mope_importance_threshold": float(
                config["mope_importance_threshold"]
            ),
            "mope_condition_dim": int(config["mope_condition_dim"]),
            "mope_top_k": config["mope_top_k"],
        },
        "method": {
            "name": (
                "explicit NVE descriptors + "
                f"{config['nve_spatial_encoder']} spatial encoder + gated K/V adapter"
                if config["prompt_generator"] == "nve_sgformer"
                else (
                    "independent HbO/HbR CNN nodes + global SGFormer + "
                    "gated K/V adapter (matched no-NVE control)"
                )
                if config["prompt_generator"] == "cnn_sgformer"
                else "fNIRS conditioned gated K/V prompt adapter"
            ),
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
            "temporal_prompt_source": (
                "disabled: no temporal CNN execution and no temporal K/V"
                if config["prompt_branch_mode"] == "spatial_only"
                else "same 36 NVE-SGFormer node tokens as the spatial stream"
                if config["prompt_stream_mode"] == "shared"
                else (
                    "36 nodes x ten overlapping three-second HbO/HbR Conv1d "
                    "tokens, one-second stride, zero-padded boundaries + "
                    "learned node IDs and sinusoidal time positions"
                )
                if config["temporal_prompt_mode"] == "overlap_3s_node_time_tokens"
                else (
                    "six anatomical regions x ten overlapping three-second "
                    "HbO/HbR Conv1d tokens, one-second stride, zero-padded "
                    "boundaries + sinusoidal region/time encodings"
                )
                if config["temporal_prompt_mode"] == "overlap_3s_region_time_tokens"
                else (
                    "ten overlapping three-second HbO/HbR Conv1d time tokens, "
                    "one-second stride, zero-padded boundaries, mean-pooled "
                    "across nodes + sinusoidal position encoding"
                )
                if config["temporal_prompt_mode"] == "overlap_3s_10_tokens"
                else (
                    "ten aligned one-second HbO/HbR Conv1d time tokens, "
                    "mean-pooled across nodes + sinusoidal position encoding"
                )
                if config["temporal_prompt_mode"] == "aligned_10_tokens"
                else "independent HbO/HbR Conv1d node tokens"
                if config["prompt_generator"] in {"nve_sgformer", "cnn_sgformer"}
                else "fNIRS node tokens"
            ),
            "spatial_prompt_source": (
                "23 explicit NVE attributes + geometry + node identity + "
                + {
                    "sgformer": "global SGFormer",
                    "two_layer_sgformer_bottleneck": (
                        "two global SGFormer layers + residual 128-64-128 bottleneck"
                    ),
                    "hybrid_sgformer": "global SGFormer + local GCN",
                    "identity": "Identity (no cross-node interaction)",
                    "two_layer_gcn": "two-layer normalized GCN",
                }[config["nve_spatial_encoder"]]
                if config["prompt_generator"] == "nve_sgformer"
                else (
                    "independent HbO/HbR CNN node features + geometry + "
                    "node identity + global SGFormer"
                )
                if config["prompt_generator"] == "cnn_sgformer"
                else "learned fNIRS prompt"
            ),
            "prompt_projection_shared": True,
            "graph_encoder": (
                (
                    {
                        "sgformer": "SGFormer global linear attention (no GCN)",
                        "two_layer_sgformer_bottleneck": (
                            "Two SGFormer global linear-attention layers with "
                            "a residual 128-64-128 bottleneck (no GCN)"
                        ),
                        "hybrid_sgformer": (
                            "SGFormer global linear attention + local GCN"
                        ),
                        "identity": "Identity (no global attention, no GCN)",
                        "two_layer_gcn": (
                            "Two-layer normalized GCN (no global attention)"
                        ),
                    }[config["nve_spatial_encoder"]]
                )
                if config["prompt_generator"] == "nve_sgformer"
                else "SGFormer global linear attention (no GCN)"
                if config["prompt_generator"] == "cnn_sgformer"
                else None
            ),
            "fnirs_backbone": (
                "explicit NVE descriptor mapper"
                if config["prompt_generator"] == "nve_sgformer"
                else "independent HbO/HbR 1D-CNN node encoder"
                if config["prompt_generator"] == "cnn_sgformer"
                else "learned fNIRS encoder"
            ),
            "hbo_hbr_separate": config["prompt_generator"] in {
                "nve_sgformer", "cnn_sgformer"
            },
            "sgformer_attention_layers": config["sgformer_attention_layers"],
            "sgformer_heads": config["sgformer_heads"],
            "sgformer_attention_residual_weight": config[
                "sgformer_attention_residual_weight"
            ],
            "sgformer_graph_weight": config["sgformer_graph_weight"],
            "prompt_generator": config["prompt_generator"],
            "prompt_components": config["prompt_components"],
            "prompt_branch_mode": config["prompt_branch_mode"],
            "nve_spatial_encoder": config["nve_spatial_encoder"],
            "temporal_prompt_mode": config["temporal_prompt_mode"],
            "mope_expert_count": int(config["mope_expert_count"]),
            "mope_temperature": float(config["mope_temperature"]),
            "mope_router_noise_std": float(config["mope_router_noise_std"]),
            "mope_importance_threshold": float(
                config["mope_importance_threshold"]
            ),
            "mope_condition_dim": int(config["mope_condition_dim"]),
            "mope_top_k": config["mope_top_k"],
            "importance_lambda": float(config["importance_lambda"]),
            "training_protocol": config["training_protocol"],
            "fnirs_control": config["fnirs_control"],
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
                "disabled and not executed"
                if config["prompt_branch_mode"] == "spatial_only"
                else "unused and frozen: temporal adapter reuses spatial NVE Prompt"
                if config["prompt_stream_mode"] == "shared"
                else (
                    "independent HbO/HbR Conv1d encoders -> overlapping 3-second "
                    "windows for every node -> [B,36,10,128] -> node-major "
                    "[B,360,128] + learned node IDs + sinusoidal time positions"
                )
                if config["temporal_prompt_mode"] == "overlap_3s_node_time_tokens"
                else (
                    "independent HbO/HbR Conv1d encoders -> six fixed regions "
                    "x ten overlapping 3-second windows -> [B,60,128] + "
                    "sinusoidal region/time positions"
                )
                if config["temporal_prompt_mode"] == "overlap_3s_region_time_tokens"
                else (
                    "independent HbO/HbR Conv1d encoders -> overlapping 3-second "
                    "windows (1-second stride, zero padding) -> node mean -> "
                    "[B,10,128] + sinusoidal position"
                )
                if config["temporal_prompt_mode"] == "overlap_3s_10_tokens"
                else (
                    "independent HbO/HbR Conv1d encoders -> ten one-second "
                    "bins -> node mean -> [B,10,128] + sinusoidal position"
                )
                if config["temporal_prompt_mode"] == "aligned_10_tokens"
                else "independent HbO/HbR Conv1d encoders; concatenate to [B,36,128]"
                if config["prompt_generator"] in {"nve_sgformer", "cnn_sgformer"}
                else "learned fNIRS node encoder"
            ),
        },
        "graph": {
            "nodes": 36,
            "edge_count": int(montage["edge_index"].shape[1]),
            "method": montage["graph_method"],
            "edges_used_by_nve_sgformer": bool(
                config["prompt_generator"] == "nve_sgformer"
                and config["nve_spatial_encoder"] in {
                    "hybrid_sgformer", "two_layer_gcn"
                }
            ),
            "edges_used_by_spatial_encoder": bool(
                config["prompt_generator"] == "nve_sgformer"
                and config["nve_spatial_encoder"] in {
                    "hybrid_sgformer", "two_layer_gcn"
                }
            ),
            "prompt_tokens": 36,
            "prompt_dimension": 200,
            "temporal_prompt_tokens": (
                360
                if config["temporal_prompt_mode"] == "overlap_3s_node_time_tokens"
                else 60
                if config["temporal_prompt_mode"] == "overlap_3s_region_time_tokens"
                else 10
                if config["temporal_prompt_mode"] in {
                    "aligned_10_tokens", "overlap_3s_10_tokens"
                }
                else 36
            ),
            "spatial_prompt_tokens": 36,
            "encoder": (
                "23-D NVE -> 128-D mapper -> "
                + (
                    {
                        "sgformer": "one-layer global SGFormer",
                        "two_layer_sgformer_bottleneck": (
                            "two-layer global SGFormer with residual 128-64-128 bottleneck"
                        ),
                        "hybrid_sgformer": (
                            "one-layer global SGFormer + one-layer local GCN"
                        ),
                        "identity": "Identity (independent nodes)",
                        "two_layer_gcn": "two-layer normalized GCN",
                    }[config["nve_spatial_encoder"]]
                )
                + (
                    "; uses 97 fixed edges"
                    if config["nve_spatial_encoder"] in {
                        "hybrid_sgformer", "two_layer_gcn"
                    }
                    else "; no GCN"
                )
                if config["prompt_generator"] == "nve_sgformer"
                else (
                    "independent HbO/HbR CNN [64+64] -> 128-D nodes -> "
                    "one-layer global SGFormer; no GCN"
                )
                if config["prompt_generator"] == "cnn_sgformer"
                else "learned fNIRS encoder"
            ),
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
        prompt_generator=config["prompt_generator"],
        mope_expert_count=int(config["mope_expert_count"]),
        mope_temperature=float(config["mope_temperature"]),
        mope_router_noise_std=float(config["mope_router_noise_std"]),
        mope_importance_threshold=float(config["mope_importance_threshold"]),
        mope_condition_dim=int(config["mope_condition_dim"]),
        mope_top_k=config["mope_top_k"],
        prompt_components=config["prompt_components"],
        prompt_branch_mode=config["prompt_branch_mode"],
        nve_spatial_encoder=config["nve_spatial_encoder"],
        temporal_prompt_mode=config["temporal_prompt_mode"],
        temporal_kv_policy=config["temporal_kv_policy"],
        temporal_future_steps=int(config["temporal_future_steps"]),
    )
    model.freeze_backbone()
    diagnostics["pretrained_load"] = load_record
    first_adapter = next(iter(model.adapters.values()))
    temporal_mask = first_adapter.temporal_attention_mask(
        query_count=10,
        prompt_count=360,
        device=torch.device("cpu"),
    )
    if temporal_mask is not None:
        allowed = ~temporal_mask
        allowed_counts = allowed.sum(dim=1)
        key_times = torch.arange(360).remainder(10)
        allowed_time_indices = [
            torch.unique(key_times[allowed[index]]).tolist()
            for index in range(10)
        ]
        diagnostics["temporal_kv_constraint"] = {
            "policy": config["temporal_kv_policy"],
            "future_steps": int(config["temporal_future_steps"]),
            "mask_shape": list(temporal_mask.shape),
            "token_order": "node-major: token_index = node * 10 + time",
            "allowed_time_indices_by_eeg_query": allowed_time_indices,
            "allowed_key_counts_by_eeg_query": allowed_counts.tolist(),
            "all_36_nodes_retained_at_each_allowed_time": bool(
                all(count % 36 == 0 for count in allowed_counts.tolist())
            ),
            "past_fnirs_times_blocked": bool(
                all(
                    all(time_index >= query_index for time_index in times)
                    for query_index, times in enumerate(allowed_time_indices)
                )
            ),
            "passed": bool(
                allowed_counts.tolist()
                == [144, 144, 144, 144, 144, 144, 144, 108, 72, 36]
            ),
        }
        if not diagnostics["temporal_kv_constraint"]["passed"]:
            raise RuntimeError(
                f"Temporal K/V mask diagnostic failed: "
                f"{diagnostics['temporal_kv_constraint']}"
            )
    diagnostics["parameters"] = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "prompt_branch_mode": config["prompt_branch_mode"],
        "nve_spatial_encoder": config["nve_spatial_encoder"],
        "disabled_temporal_branch_trainable": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and (
                "hbo_temporal_encoder" in name
                or "hbr_temporal_encoder" in name
                or ".temporal_k." in name
                or ".temporal_v." in name
                or ".cross_attention_t." in name
                or name.endswith("temporal_gate")
            )
        ),
        "frozen_backbone": sum(
            parameter.numel() for parameter in model.backbone.parameters()
        ),
        "graph_prompt": sum(
            parameter.numel() for parameter in model.graph_prompt.parameters()
        ),
        "spatial_encoder": sum(
            parameter.numel()
            for parameter in model.graph_prompt.sgformer.parameters()
        ) if getattr(model.graph_prompt, "sgformer", None) is not None else 0,
        "sgformer": (
            sum(
                parameter.numel()
                for parameter in model.graph_prompt.sgformer.parameters()
            )
            if config["nve_spatial_encoder"] in {
                "sgformer", "two_layer_sgformer_bottleneck", "hybrid_sgformer"
            } else 0
        ),
        "local_gcn": (
            sum(
                parameter.numel()
                for parameter in model.graph_prompt.sgformer.local_gcn.parameters()
            )
            if config["nve_spatial_encoder"] == "hybrid_sgformer" else 0
        ),
        "two_layer_gcn": (
            sum(
                parameter.numel()
                for parameter in model.graph_prompt.sgformer.parameters()
            )
            if config["nve_spatial_encoder"] == "two_layer_gcn" else 0
        ),
        "fnirs_t": sum(
            parameter.numel()
            for parameter in model.graph_prompt.fnirs_t.parameters()
        ) if getattr(model.graph_prompt, "fnirs_t", None) is not None else 0,
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
        "mope_router": (
            sum(
                parameter.numel()
                for parameter in model.graph_prompt.router.parameters()
            )
            if config["prompt_generator"] == "mope"
            else None
        ),
        "mope_experts": (
            model.graph_prompt.prompt_experts.numel()
            if config["prompt_generator"] == "mope"
            else None
        ),
        "mope_static_prompt": (
            model.graph_prompt.static_prompt.numel()
            if config["prompt_generator"] == "mope"
            else None
        ),
        "mope_component_gates": (
            sum(
                parameter.numel()
                for name, parameter in model.graph_prompt.named_parameters()
                if "gate" in name
            )
            if config["prompt_generator"] == "mope"
            else None
        ),
    }
    diagnostics["initial_gates"] = model.gate_values()
    diagnostics["initial_prompt_component_gates"] = (
        model.prompt_component_gates()
    )
    write_json(output_dir / "diagnostics.json", diagnostics)

    smoke_device = torch.device("cpu")
    model.to(smoke_device)
    eeg, fnirs, smoke_target, _ = next(iter(loaders["train"]))
    model.eval()
    if config["prompt_generator"] == "mope":
        with torch.no_grad():
            fnirs_t_smoke = model.graph_prompt.fnirs_t(
                fnirs[:1].float().permute(0, 2, 1, 3).contiguous()
            )
        diagnostics["fnirs_t_encoder_smoke"] = {
            "input_order": ["HbO", "HbR"],
            "output_shape": list(fnirs_t_smoke.shape),
            "hbo_hbr_separate": False,
            "mixing": "both Conv2d stems have in_channels=2",
            "finite_outputs_passed": bool(torch.isfinite(fnirs_t_smoke).all()),
        }
    if config["prompt_generator"] == "nve_sgformer":
        with torch.no_grad():
            descriptor_smoke = model.graph_prompt._event_descriptors(fnirs[:1].float())
        diagnostics["nve_descriptor_smoke"] = {
            "definition": "explicit node-wise neurovascular event attributes",
            "names": list(model.graph_prompt.DESCRIPTOR_NAMES),
            "shape": list(descriptor_smoke.shape),
            "count": len(model.graph_prompt.DESCRIPTOR_NAMES),
            "finite_outputs_passed": bool(torch.isfinite(descriptor_smoke).all()),
            "uses_labels": False,
            "uses_eeg": False,
            "uses_test_fitted_statistics": False,
        }
    if config["temporal_prompt_mode"] in {
        "aligned_10_tokens", "overlap_3s_10_tokens",
        "overlap_3s_region_time_tokens", "overlap_3s_node_time_tokens",
    }:
        temporal_input = fnirs[:1].float()
        batch, nodes, _, samples = temporal_input.shape
        with torch.no_grad():
            hbo_signal = temporal_input[:, :, 0, :].reshape(
                batch * nodes, 1, samples
            )
            hbr_signal = temporal_input[:, :, 1, :].reshape(
                batch * nodes, 1, samples
            )
            if config["temporal_prompt_mode"] in {
                "overlap_3s_10_tokens", "overlap_3s_region_time_tokens",
                "overlap_3s_node_time_tokens",
            }:
                hbo_windows = model.graph_prompt._overlapping_three_second_windows(
                    hbo_signal
                )
                hbr_windows = model.graph_prompt._overlapping_three_second_windows(
                    hbr_signal
                )
                window_samples = hbo_windows.shape[-1]
                hbo_time_features = model.graph_prompt.hbo_temporal_encoder(
                    hbo_windows.reshape(batch * nodes * 10, 1, window_samples)
                )
                hbr_time_features = model.graph_prompt.hbr_temporal_encoder(
                    hbr_windows.reshape(batch * nodes * 10, 1, window_samples)
                )
            else:
                hbo_windows = None
                hbr_windows = None
                hbo_time_features = model.graph_prompt.hbo_temporal_encoder(hbo_signal)
                hbr_time_features = model.graph_prompt.hbr_temporal_encoder(hbr_signal)
            base_time_prompt = model.graph_prompt(temporal_input)[1]
            perturbed_time_input = temporal_input.clone()
            if config["temporal_prompt_mode"] in {
                "overlap_3s_region_time_tokens", "overlap_3s_node_time_tokens"
            }:
                perturbed_time_input[:, 0, :, 20:30] += 0.5
            else:
                perturbed_time_input[..., 20:30] += 0.5
            perturbed_time_prompt = model.graph_prompt(perturbed_time_input)[1]
        temporal_perturbation = (
            perturbed_time_prompt - base_time_prompt
        ).abs()
        affected_time_tokens = torch.nonzero(
            temporal_perturbation.amax(dim=(0, 2)) > 1e-7,
            as_tuple=False,
        ).flatten().tolist()
        overlap_mode = config["temporal_prompt_mode"] in {
            "overlap_3s_10_tokens", "overlap_3s_region_time_tokens",
            "overlap_3s_node_time_tokens",
        }
        region_time_mode = (
            config["temporal_prompt_mode"] == "overlap_3s_region_time_tokens"
        )
        node_time_mode = (
            config["temporal_prompt_mode"] == "overlap_3s_node_time_tokens"
        )
        region_names = list(model.graph_prompt.TEMPORAL_REGION_NAMES)
        region_nodes = [
            list(region)
            for region in model.graph_prompt.TEMPORAL_REGION_NODE_INDICES
        ]
        diagnostics["true_temporal_token_smoke"] = {
            "fnirs_sampling_rate_hz": 10,
            "trial_samples": int(samples),
            "time_token_count": 10,
            "region_count": 6 if region_time_mode else None,
            "node_count": 36 if node_time_mode else None,
            "temporal_prompt_token_count": (
                360 if node_time_mode else 60 if region_time_mode else 10
            ),
            "seconds_per_token": 1.0,
            "aggregation_window_seconds": (
                3.0
                if overlap_mode
                else 1.0
            ),
            "window_stride_seconds": 1.0,
            "boundary_padding": (
                "one second of zeros on both ends"
                if overlap_mode
                else "none"
            ),
            "node_aggregation": (
                "fixed mean within each of six anatomical regions"
                if region_time_mode
                else "none: all 36 nodes remain separate"
                if node_time_mode
                else "fixed mean across 36 nodes"
            ),
            "region_names": region_names if region_time_mode else None,
            "region_node_indices_zero_based": region_nodes if region_time_mode else None,
            "region_sizes": (
                [len(region) for region in region_nodes]
                if region_time_mode else None
            ),
            "all_nodes_covered_exactly_once": (
                sorted(node for region in region_nodes for node in region)
                == list(range(36))
                if region_time_mode else None
            ),
            "position_encoding": (
                "fixed sinusoidal region + time, no trainable parameters"
                if region_time_mode
                else "learned node ID + fixed sinusoidal time"
                if node_time_mode
                else "fixed sinusoidal time, no trainable parameters"
            ),
            "hbo_encoder_output_shape": list(hbo_time_features.shape),
            "hbr_encoder_output_shape": list(hbr_time_features.shape),
            "raw_window_shape": (
                list(hbo_windows.shape) if hbo_windows is not None else None
            ),
            "temporal_prompt_shape": list(base_time_prompt.shape),
            "perturbed_input_samples": [20, 30],
            "perturbed_node_indices_zero_based": (
                [0] if region_time_mode or node_time_mode else list(range(36))
            ),
            "affected_temporal_token_indices_zero_based": affected_time_tokens,
            "expected_affected_token_indices_zero_based": (
                [1, 2, 3] if overlap_mode else None
            ),
            "overlap_locality_passed": (
                affected_time_tokens == [1, 2, 3] if overlap_mode else None
            ),
            "front_padding_is_zero": (
                bool(torch.count_nonzero(hbo_windows[..., 0, :10]).item() == 0)
                if overlap_mode else None
            ),
            "back_padding_is_zero": (
                bool(torch.count_nonzero(hbo_windows[..., -1, -10:]).item() == 0)
                if overlap_mode else None
            ),
            "max_abs_prompt_change": float(temporal_perturbation.max()),
            "finite_outputs_passed": bool(
                torch.isfinite(base_time_prompt).all()
                and torch.isfinite(perturbed_time_prompt).all()
            ),
            "time_sensitive_passed": bool(
                temporal_perturbation.max().item() > 1e-7
            ),
            "shape_passed": bool(
                tuple(base_time_prompt.shape[1:])
                == (
                    (360, 200) if node_time_mode
                    else (60, 200) if region_time_mode
                    else (10, 200)
                )
                and (
                    (
                        tuple(hbo_windows.shape[1:]) == (1, 10, 30)
                        and tuple(hbr_windows.shape[1:]) == (1, 10, 30)
                        and tuple(hbo_time_features.shape[1:]) == (64, 1)
                        and tuple(hbr_time_features.shape[1:]) == (64, 1)
                    )
                    if overlap_mode
                    else (
                        tuple(hbo_time_features.shape[1:]) == (64, 10)
                        and tuple(hbr_time_features.shape[1:]) == (64, 10)
                    )
                )
            ),
        }
    if config["prompt_generator"] == "cnn_sgformer":
        cnn_input = fnirs[:1].float()
        batch, nodes, _, samples = cnn_input.shape
        with torch.no_grad():
            hbo_features = model.graph_prompt.hbo_temporal_encoder(
                cnn_input[:, :, 0, :].reshape(batch * nodes, 1, samples)
            ).squeeze(-1)
            hbr_features = model.graph_prompt.hbr_temporal_encoder(
                cnn_input[:, :, 1, :].reshape(batch * nodes, 1, samples)
            ).squeeze(-1)
            cnn_nodes = torch.cat((hbo_features, hbr_features), dim=-1).reshape(
                batch, nodes, -1
            )
        hbo_parameter_ids = {
            parameter.data_ptr()
            for parameter in model.graph_prompt.hbo_temporal_encoder.parameters()
        }
        hbr_parameter_ids = {
            parameter.data_ptr()
            for parameter in model.graph_prompt.hbr_temporal_encoder.parameters()
        }
        diagnostics["cnn_node_feature_smoke"] = {
            "input_order": ["HbO", "HbR"],
            "hbo_feature_shape": list(hbo_features.shape),
            "hbr_feature_shape": list(hbr_features.shape),
            "concatenated_node_shape": list(cnn_nodes.shape),
            "hbo_hbr_parameter_overlap": len(
                hbo_parameter_ids.intersection(hbr_parameter_ids)
            ),
            "independent_parameters_passed": bool(
                not hbo_parameter_ids.intersection(hbr_parameter_ids)
            ),
            "uses_explicit_nve_descriptors": False,
            "finite_outputs_passed": bool(torch.isfinite(cnn_nodes).all()),
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
    if (
        config["prompt_generator"] == "nve_sgformer"
        and config["nve_spatial_encoder"]
        == "two_layer_sgformer_bottleneck"
    ):
        encoder = model.graph_prompt.sgformer
        layer1_parameter_ids = {
            parameter.data_ptr() for parameter in encoder.layer1.parameters()
        }
        layer2_parameter_ids = {
            parameter.data_ptr() for parameter in encoder.layer2.parameters()
        }
        with torch.no_grad():
            nve_input = fnirs[:1].float()
            descriptors = model.graph_prompt._event_descriptors(nve_input)
            node_count = nve_input.shape[1]
            nve_nodes = (
                model.graph_prompt.nve_projection(descriptors)
                + model.graph_prompt.geometry_encoder(
                    model.graph_prompt.positions_3d
                ).unsqueeze(0)
                + model.graph_prompt.node_embedding(
                    torch.arange(node_count, device=nve_input.device)
                ).unsqueeze(0)
            )
            first_layer_output = encoder.layer1(nve_nodes)
            bridged_output = encoder.bottleneck_output_norm(
                first_layer_output + encoder.bottleneck(first_layer_output)
            )
            second_layer_output = encoder.layer2(bridged_output)
            perturbed_input = nve_input.clone()
            perturbed_input[:, 0, :, :] += 0.5
            perturbed_spatial = model.graph_prompt(perturbed_input)[0]
        bottleneck_difference = (bridged_output - first_layer_output).abs()
        second_layer_difference = (second_layer_output - bridged_output).abs()
        propagated_node_difference = (
            perturbed_spatial - smoke["spatial_prompt"]
        ).abs().amax(dim=-1)[0]
        module_class_names = {
            module.__class__.__name__ for module in encoder.modules()
        }
        diagnostics["two_layer_sgformer_bottleneck_smoke"] = {
            "attention_layer_count": 2,
            "model_dimension": int(config["graph_dimension"]),
            "bottleneck_dimension": int(encoder.bottleneck_dimension),
            "bottleneck_path": [128, 64, 128],
            "bottleneck_is_residual": True,
            "layer1_output_shape": list(first_layer_output.shape),
            "bottleneck_output_shape": list(bridged_output.shape),
            "layer2_output_shape": list(second_layer_output.shape),
            "layer_parameters_are_independent": bool(
                layer1_parameter_ids.isdisjoint(layer2_parameter_ids)
            ),
            "bottleneck_changes_representation": bool(
                bottleneck_difference.max().item() > 1e-7
            ),
            "second_layer_changes_representation": bool(
                second_layer_difference.max().item() > 1e-7
            ),
            "perturbed_node": 0,
            "changed_node_count_after_global_attention": int(
                (propagated_node_difference > 1e-7).sum().item()
            ),
            "cross_node_propagation_passed": bool(
                (propagated_node_difference[1:] > 1e-7).any()
            ),
            "contains_gcn": bool("DenseGraphConv" in module_class_names),
            "passed": bool(
                tuple(second_layer_output.shape[1:]) == (36, 128)
                and layer1_parameter_ids.isdisjoint(layer2_parameter_ids)
                and bottleneck_difference.max().item() > 1e-7
                and second_layer_difference.max().item() > 1e-7
                and (propagated_node_difference[1:] > 1e-7).any()
                and "DenseGraphConv" not in module_class_names
            ),
        }
    if (
        config["prompt_generator"] == "nve_sgformer"
        and config["nve_spatial_encoder"] == "hybrid_sgformer"
    ):
        original_adjacency = model.graph_prompt.adjacency.detach().clone()
        identity_adjacency = torch.eye(
            original_adjacency.shape[0],
            dtype=original_adjacency.dtype,
            device=original_adjacency.device,
        )
        with torch.no_grad():
            model.graph_prompt.adjacency.copy_(identity_adjacency)
            no_edge_spatial = model.graph_prompt(fnirs[:1].float())[0]
            model.graph_prompt.adjacency.copy_(original_adjacency)
        adjacency_difference = (
            smoke["spatial_prompt"] - no_edge_spatial
        ).abs()
        diagnostics["hybrid_graph_adjacency_sensitivity_smoke"] = {
            "fixed_edge_count": int(montage["edge_index"].shape[1]),
            "graph_weight": float(config["sgformer_graph_weight"]),
            "original_adjacency_shape": list(original_adjacency.shape),
            "mean_abs_prompt_difference_without_edges": float(
                adjacency_difference.mean().item()
            ),
            "max_abs_prompt_difference_without_edges": float(
                adjacency_difference.max().item()
            ),
            "adjacency_restored": bool(torch.equal(
                model.graph_prompt.adjacency, original_adjacency
            )),
            "edges_change_spatial_prompt": bool(
                adjacency_difference.max().item() > 1e-7
            ),
            "passed": bool(
                adjacency_difference.max().item() > 1e-7
                and torch.equal(model.graph_prompt.adjacency, original_adjacency)
            ),
        }
    elif (
        config["prompt_generator"] == "nve_sgformer"
        and config["nve_spatial_encoder"] == "identity"
    ):
        perturbed_fnirs = fnirs[:1].float().clone()
        perturbed_fnirs[:, 0, :, :] += 0.5
        with torch.no_grad():
            perturbed_spatial = model.graph_prompt(perturbed_fnirs)[0]
        target_difference = (
            perturbed_spatial[:, 0] - smoke["spatial_prompt"][:, 0]
        ).abs()
        other_difference = (
            perturbed_spatial[:, 1:] - smoke["spatial_prompt"][:, 1:]
        ).abs()
        diagnostics["identity_cross_node_independence_smoke"] = {
            "perturbed_node": 0,
            "target_node_max_abs_difference": float(target_difference.max()),
            "other_nodes_max_abs_difference": float(other_difference.max()),
            "target_node_changed": bool(target_difference.max() > 1e-7),
            "other_nodes_unchanged": bool(other_difference.max() <= 1e-7),
            "passed": bool(
                target_difference.max() > 1e-7
                and other_difference.max() <= 1e-7
            ),
        }
    elif (
        config["prompt_generator"] == "nve_sgformer"
        and config["nve_spatial_encoder"] == "two_layer_gcn"
    ):
        perturbed_fnirs = fnirs[:1].float().clone()
        perturbed_fnirs[:, 0, :, :] += 0.5
        with torch.no_grad():
            perturbed_spatial = model.graph_prompt(perturbed_fnirs)[0]
        node_differences = (
            perturbed_spatial - smoke["spatial_prompt"]
        ).abs().amax(dim=-1)[0]
        adjacency = model.graph_prompt.adjacency
        two_hop_mask = (adjacency @ adjacency) > 0
        outside_two_hop = node_differences[~two_hop_mask[0]]
        changed_nodes = node_differences > 1e-7
        diagnostics["gcn_cross_node_propagation_smoke"] = {
            "perturbed_node": 0,
            "changed_node_count": int(changed_nodes.sum()),
            "non_target_node_max_abs_difference": float(
                node_differences[1:].max()
            ),
            "outside_two_hop_max_abs_difference": float(
                outside_two_hop.max() if outside_two_hop.numel() else 0.0
            ),
            "propagates_beyond_target": bool(changed_nodes[1:].any()),
            "limited_to_two_hops": bool(
                not outside_two_hop.numel()
                or outside_two_hop.max() <= 1e-7
            ),
            "passed": bool(
                changed_nodes[1:].any()
                and (
                    not outside_two_hop.numel()
                    or outside_two_hop.max() <= 1e-7
                )
            ),
        }
    same_prompt_shape = (
        smoke["spatial_prompt"].shape == smoke["temporal_prompt"].shape
    )
    if same_prompt_shape:
        prompt_difference = (
            smoke["spatial_prompt"] - smoke["temporal_prompt"]
        ).abs()
    else:
        # Token counts intentionally differ for true temporal tokens. Compare
        # stream-level means while recording the shape difference explicitly.
        prompt_difference = (
            smoke["spatial_prompt"].mean(dim=1)
            - smoke["temporal_prompt"].mean(dim=1)
        ).abs()
    diagnostics["prompt_stream_smoke"] = {
        "mode": config["prompt_stream_mode"],
        "spatial_prompt_shape": list(smoke["spatial_prompt"].shape),
        "temporal_prompt_shape": list(smoke["temporal_prompt"].shape),
        "same_token_shape": bool(same_prompt_shape),
        "mean_abs_difference": float(prompt_difference.mean().item()),
        "max_abs_difference": float(prompt_difference.max().item()),
        "distinct_content_passed": bool(prompt_difference.max().item() > 0.0),
        "expected_relation": (
            "identical" if config["prompt_stream_mode"] == "shared" else "distinct"
        ),
        "expected_relation_passed": bool(
            prompt_difference.max().item() == 0.0
            if config["prompt_stream_mode"] == "shared"
            else (not same_prompt_shape or prompt_difference.max().item() > 0.0)
        ),
        "finite_outputs_passed": bool(
            torch.isfinite(smoke["spatial_prompt"]).all()
            and torch.isfinite(smoke["temporal_prompt"]).all()
        ),
        "temporal_prompt_zero_for_spatial_only": bool(
            config["prompt_branch_mode"] != "spatial_only"
            or torch.count_nonzero(smoke["temporal_prompt"]).item() == 0
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
    patience = config["early_stopping_patience"]
    started = time.time()
    stage1_summary: dict[str, Any] | None = None
    if config["training_protocol"] == "prompt_only":
        # Stage 1: EEG-only head pretraining.  The prompts stay frozen at
        # their zero gates, so Fusion is identical to EEG and the shared
        # all-patch classifier is learned from the frozen backbone alone.
        stage1_source = config.get("stage1_checkpoint_source")
        if stage1_source:
            stage1_checkpoint = torch.load(
                Path(stage1_source), map_location=device, weights_only=False
            )
            source_state = stage1_checkpoint.get("model", stage1_checkpoint)
            classifier_state = {
                key: value for key, value in source_state.items()
                if key.startswith("classifier.")
            }
            expected_classifier_keys = {
                key for key in model.state_dict() if key.startswith("classifier.")
            }
            if set(classifier_state) != expected_classifier_keys:
                raise RuntimeError(
                    "Stage-1 source does not contain the exact shared classifier state"
                )
            load_result = model.load_state_dict(classifier_state, strict=False)
            if load_result.unexpected_keys:
                raise RuntimeError(f"Unexpected stage-1 keys: {load_result.unexpected_keys}")
            stage1 = None
            stage1_history = []
            stage1_summary = {
                "reused": True,
                "source": str(Path(stage1_source).resolve()),
                "source_epoch": stage1_checkpoint.get("epoch"),
                "classifier_keys_loaded": len(classifier_state),
            }
        else:
            set_trainable_stage(model, "head")
            optimizer = make_optimizer(model, config, "head")
            stage1 = run_stage(
                model, loaders, optimizer, config, output_dir,
                "stage1_eeg_head", "stage1_head.pt",
                int(config["stage1_epochs"]),
                {**config["loss_weights"], "fnirs": 0.0},
                None, started,
            )
            stage1_checkpoint = torch.load(
                output_dir / "stage1_head.pt",
                map_location=device, weights_only=False,
            )
            model.load_state_dict(stage1_checkpoint["model"])
            stage1_history = stage1["history"]
            stage1_summary = {
                "reused": False,
                "epochs": len(stage1["history"]),
                "best_epoch": stage1["best_epoch"],
                "best_val": stage1["best_val"],
                "checkpoint": str(output_dir / "stage1_head.pt"),
            }
        # Stage 2: prompt-only.  The shared classifier stays frozen; only the
        # fNIRS graph prompt, adapters, component gates and the small fNIRS
        # head are trained.
        set_trainable_stage(model, "prompt_only")
        optimizer = make_optimizer(model, config, "prompt_only")
        stage2 = run_stage(
            model, loaders, optimizer, config, output_dir,
            "stage2_prompt_only", "best.pt",
            int(config["epochs"]), config["loss_weights"],
            patience, started,
        )
        history = stage1_history + stage2["history"]
        best_epoch = stage2["best_epoch"]
        best_val = stage2["best_val"]
        stopped_early = stage2["stopped_early"]
        stop_epoch = stage2["stop_epoch"]
    else:
        set_trainable_stage(model, "joint")
        optimizer = make_optimizer(model, config, "joint")
        joint = run_stage(
            model, loaders, optimizer, config, output_dir,
            "frozen_cbramod_split_temporal_cnn_spatial_sgformer_gated_kv_prompt",
            "best.pt", int(config["epochs"]), config["loss_weights"],
            patience, started,
        )
        history = joint["history"]
        best_epoch = joint["best_epoch"]
        best_val = joint["best_val"]
        stopped_early = joint["stopped_early"]
        stop_epoch = joint["stop_epoch"]
    write_json(output_dir / "history.json", history)

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
        "model": (
            "CBraMod + explicit NVE + "
            + {
                "sgformer": "global SGFormer",
                "two_layer_sgformer_bottleneck": (
                    "two-layer global SGFormer + residual 128-64-128 bottleneck"
                ),
                "hybrid_sgformer": "global SGFormer + local GCN",
                "identity": "node-wise Identity",
                "two_layer_gcn": "two-layer normalized GCN",
            }[config["nve_spatial_encoder"]]
            + " gated K/V prompts"
            if config["prompt_generator"] == "nve_sgformer"
            else (
                "CBraMod + independent HbO/HbR CNN nodes + global SGFormer "
                "+ gated K/V prompts (matched no-NVE control)"
            )
            if config["prompt_generator"] == "cnn_sgformer"
            else "CBraMod + fNIRS gated K/V prompts"
        ),
        "task": config["task"],
        "task_name": TASKS[config["task"]]["name"],
        "modalities": ["eeg", "hbo", "hbr"],
        "seed": config["seed"],
        "splits": split_subjects,
        "backbone_frozen_all_epochs": True,
        "prompt_layer_indices": config["prompt_layer_indices"],
        "graph_encoder": (
            (
                {
                    "sgformer": "SGFormer global linear attention without GCN",
                    "two_layer_sgformer_bottleneck": (
                        "Two SGFormer global linear-attention layers with a "
                        "residual 128-64-128 bottleneck and no GCN"
                    ),
                    "hybrid_sgformer": (
                        "SGFormer global linear attention + local normalized GCN"
                    ),
                    "identity": "Identity without cross-node interaction",
                    "two_layer_gcn": (
                        "Two-layer normalized GCN without global attention"
                    ),
                }[config["nve_spatial_encoder"]]
            )
            if config["prompt_generator"] == "nve_sgformer"
            else "SGFormer global linear attention without GCN"
            if config["prompt_generator"] == "cnn_sgformer"
            else None
        ),
        "fnirs_backbone": (
            "explicit 23-D NVE attributes"
            if config["prompt_generator"] == "nve_sgformer"
            else "independent HbO/HbR 1D-CNN node encoder"
            if config["prompt_generator"] == "cnn_sgformer"
            else "learned"
        ),
        "chromophore_encoder_mode": config["chromophore_encoder_mode"],
        "prompt_stream_mode": config["prompt_stream_mode"],
        "prompt_branch_mode": config["prompt_branch_mode"],
        "nve_spatial_encoder": config["nve_spatial_encoder"],
        "temporal_prompt_mode": config["temporal_prompt_mode"],
        "temporal_kv_policy": config["temporal_kv_policy"],
        "temporal_future_steps": int(config["temporal_future_steps"]),
        "temporal_prompt_source": (
            "disabled: strict spatial-only ablation"
            if config["prompt_branch_mode"] == "spatial_only"
            else "shared 36-token NVE-SGFormer spatial Prompt"
            if config["prompt_stream_mode"] == "shared"
            else (
                "36 nodes x 10 overlapping 3-second HbO/HbR CNN windows "
                "with 1-second stride and zero-padded boundaries + learned "
                "node IDs and sinusoidal time positions"
            )
            if config["temporal_prompt_mode"] == "overlap_3s_node_time_tokens"
            else (
                "6 anatomical regions x 10 overlapping 3-second HbO/HbR CNN "
                "windows with 1-second stride and zero-padded boundaries + "
                "sinusoidal region/time positions"
            )
            if config["temporal_prompt_mode"] == "overlap_3s_region_time_tokens"
            else (
                "10 overlapping 3-second HbO/HbR CNN windows with 1-second "
                "stride and zero-padded boundaries, pooled across nodes + "
                "sinusoidal temporal positions"
            )
            if config["temporal_prompt_mode"] == "overlap_3s_10_tokens"
            else (
                "10 aligned one-second HbO/HbR CNN tokens pooled across nodes "
                "+ sinusoidal temporal positions"
            )
            if config["temporal_prompt_mode"] == "aligned_10_tokens"
            else "independent HbO/HbR Conv1d tokens"
            if config["prompt_generator"] in {"nve_sgformer", "cnn_sgformer"}
            else "learned tokens"
        ),
        "spatial_prompt_source": (
            "23-D NVE + geometry + node identity + "
            + {
                "sgformer": "global SGFormer",
                "two_layer_sgformer_bottleneck": (
                    "two global SGFormer layers + residual bottleneck"
                ),
                "hybrid_sgformer": "global SGFormer + local GCN",
                "identity": "Identity",
                "two_layer_gcn": "two-layer normalized GCN",
            }[config["nve_spatial_encoder"]]
            if config["prompt_generator"] == "nve_sgformer"
            else (
                "independent HbO/HbR CNN node features + geometry + "
                "node identity + global SGFormer"
            )
            if config["prompt_generator"] == "cnn_sgformer"
            else "learned prompt"
        ),
        "hbo_hbr_separate": config["prompt_generator"] in {
            "nve_sgformer", "cnn_sgformer"
        },
        "spatial_feature_source": (
            "explicit_23d_nve_descriptors"
            if config["prompt_generator"] == "nve_sgformer"
            else "independent_hbo_hbr_cnn_features"
            if config["prompt_generator"] == "cnn_sgformer"
            else "learned_prompt"
        ),
        "uses_explicit_nve_descriptors": bool(
            config["prompt_generator"] == "nve_sgformer"
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
        "best_prompt_component_gates": model.prompt_component_gates(),
        "final_prompt_component_gates": (
            model.prompt_component_gates()
        ),
        "routing_statistics_best": next(
            (
                record.get("routing_statistics", {})
                for record in history
                if record.get("is_best")
            ),
            {},
        ),
        "prompt_generator": config["prompt_generator"],
        "prompt_components": config["prompt_components"],
        "prompt_branch_mode": config["prompt_branch_mode"],
        "training_protocol": config["training_protocol"],
        "fnirs_control": config["fnirs_control"],
        "importance_lambda": float(config["importance_lambda"]),
        "mope_top_k": config["mope_top_k"],
        "mope_expert_count": int(config["mope_expert_count"]),
        "mope_temperature": float(config["mope_temperature"]),
        "mope_router_noise_std": float(config["mope_router_noise_std"]),
        "mope_importance_threshold": float(
            config["mope_importance_threshold"]
        ),
        "stage1": stage1_summary,
        "nve_descriptor_names": (
            list(model.graph_prompt.DESCRIPTOR_NAMES)
            if config["prompt_generator"] == "nve_sgformer" else None
        ),
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
