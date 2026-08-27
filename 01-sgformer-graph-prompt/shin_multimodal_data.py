"""Strictly aligned SHIN EEG-HbO/HbR trial loader.

This is a new SHIN adapter for the fNIRS-graph-prompt experiment.  It does not
modify the historical DAMFNet or LaBraM baseline adapters.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, lfilter
from scipy.spatial import Delaunay, QhullError
import torch
from torch.utils.data import Dataset


SHIN_EEG_CHANNELS = [
    "F7", "AFF5h", "F3", "AFp1", "AFp2", "AFF6h", "F4", "F8",
    "AFF1h", "AFF2h", "Cz", "Pz", "FCC5h", "FCC3h", "CCP5h",
    "CCP3h", "T7", "P7", "P3", "PPO1h", "POO1", "POO2", "PPO2h",
    "P4", "FCC4h", "FCC6h", "CCP4h", "CCP6h", "P8", "T8",
]

LABRAM_CHANNELS = [
    "F7", "AF5", "F3", "FP1", "FP2", "AF6", "F4", "F8", "AF1",
    "AF2", "CZ", "PZ", "FC5", "FC3", "CP5", "CP3", "T7", "P7",
    "P3", "PO1", "O1", "O2", "PO2", "P4", "FC4", "FC6", "CP4",
    "CP6", "P8", "T8",
]

TASKS = {
    "mi": {
        "name": "EEG-fNIRS-MI",
        "description": "left_hand (0) vs right_hand (1)",
        "session_indices": (0, 2, 4),
        "session_names": ("ses-0imagery", "ses-2imagery", "ses-4imagery"),
        "labels": {"left_hand": 0, "right_hand": 1},
    },
    "ma": {
        "name": "EEG-fNIRS-MA",
        "description": "subtraction (0) vs rest (1)",
        "session_indices": (1, 3, 5),
        "session_names": ("ses-1arithmetic", "ses-3arithmetic", "ses-5arithmetic"),
        "labels": {"subtraction": 0, "rest": 1},
    },
}

# MNE/OMLC absorption coefficients at 760 and 850 nm, columns HbO/HbR.
ABSORPTION = np.asarray(
    [[134.9558, 356.624156], [243.6574, 159.210996]],
    dtype=np.float64,
)


@dataclass(frozen=True)
class SplitArrays:
    eeg_uv: np.ndarray
    fnirs_um: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    sessions: np.ndarray
    trials: np.ndarray
    keys: np.ndarray
    details: list[dict]


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _exactly_one(folder: Path, pattern: str) -> Path:
    paths = list(folder.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"{folder}: expected one {pattern}, found {len(paths)}")
    return paths[0]


def _validate_fnirs_channels(clab: object) -> list[str]:
    names = [str(value) for value in np.asarray(clab).reshape(-1)]
    if len(names) != 72:
        raise ValueError(f"Expected 72 wavelength channels, got {len(names)}")
    low = [name.removesuffix("lowWL") for name in names[:36]]
    high = [name.removesuffix("highWL") for name in names[36:]]
    if any(not name.endswith("lowWL") for name in names[:36]):
        raise ValueError("First 36 channels are not low-wavelength channels")
    if any(not name.endswith("highWL") for name in names[36:]):
        raise ValueError("Last 36 channels are not high-wavelength channels")
    if low != high:
        raise ValueError("Low/high wavelength source-detector orders differ")
    return low


def intensity_to_hbo_hbr(
    intensity: np.ndarray,
    distance_m: float = 0.03,
    ppf: float = 6.0,
) -> np.ndarray:
    """Return HbO/HbR concentrations as [time, node, 2] in micromolar."""
    intensity = np.asarray(intensity, dtype=np.float64)
    if intensity.ndim != 2 or intensity.shape[1] != 72:
        raise ValueError(f"Expected intensity [time,72], got {intensity.shape}")
    safe = np.abs(intensity)
    for channel in range(safe.shape[1]):
        positive = safe[:, channel][safe[:, channel] > 0]
        if not len(positive):
            raise ValueError(f"Intensity channel {channel} has no positive samples")
        safe[:, channel] = np.maximum(safe[:, channel], positive.min())
    optical_density = -np.log(safe / safe.mean(axis=0, keepdims=True))
    inverse = np.linalg.pinv(ABSORPTION * distance_m * ppf) * 1e-3
    concentration = np.einsum(
        "ab,tcb->tca",
        inverse,
        np.stack((optical_density[:, :36], optical_density[:, 36:]), axis=-1),
    )
    return concentration * 1e6


def preprocess_eeg_cbramod(data_uv: np.ndarray) -> np.ndarray:
    """Match the established CBraMod SHIN EEG preprocessing."""
    data_uv = data_uv - data_uv.mean(axis=0, keepdims=True)
    b, a = butter(5, [0.3 / 100.0, 50.0 / 100.0], btype="band")
    return lfilter(b, a, data_uv, axis=-1).astype(np.float32)


def load_fnirs_montage(fnirs_root: Path, subject: int = 1) -> dict:
    """Load the 36 source-detector channel midpoints and deterministic graph."""
    path = fnirs_root / f"subject {subject:02d}" / "mnt.mat"
    montage = loadmat(path, simplify_cells=True)["mnt"]
    positions = np.asarray(montage["pos_3d"], dtype=np.float64).T
    xy = np.column_stack((
        np.asarray(montage["x"], dtype=np.float64),
        np.asarray(montage["y"], dtype=np.float64),
    ))
    labels = [str(value) for value in np.asarray(montage["clab"]).reshape(-1)]
    sd = np.asarray(montage["sd"], dtype=np.int64)
    if positions.shape != (36, 3) or xy.shape != (36, 2) or sd.shape != (36, 2):
        raise RuntimeError(
            f"Unexpected montage shapes: pos={positions.shape}, xy={xy.shape}, sd={sd.shape}"
        )
    if not np.isfinite(positions).all() or not np.isfinite(xy).all():
        raise RuntimeError(f"Non-finite fNIRS montage coordinates in {path}")

    edge_set: set[tuple[int, int]] = set()
    graph_method = "delaunay_2d"
    try:
        simplices = Delaunay(xy).simplices
        for triangle in simplices:
            for left, right in ((0, 1), (1, 2), (2, 0)):
                a, b = sorted((int(triangle[left]), int(triangle[right])))
                edge_set.add((a, b))
    except QhullError:
        graph_method = "knn_3d_fallback_k4"
        distances = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=-1
        )
        for node in range(36):
            for neighbor in np.argsort(distances[node])[1:5]:
                a, b = sorted((node, int(neighbor)))
                edge_set.add((a, b))

    # Channels sharing an optode are physiologically adjacent even when the
    # 2-D triangulation misses the relationship.
    sd_zero = sd - 1 if sd.min() == 1 else sd.copy()
    for i in range(36):
        for j in range(i + 1, 36):
            same_source = sd_zero[i, 0] == sd_zero[j, 0]
            same_detector = sd_zero[i, 1] == sd_zero[j, 1]
            if same_source or same_detector:
                edge_set.add((i, j))
    edge_index = np.asarray(sorted(edge_set), dtype=np.int64).T
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise RuntimeError("Failed to construct fNIRS graph edges")
    return {
        "path": str(path.resolve()),
        "positions_3d": positions.astype(np.float32),
        "positions_2d": xy.astype(np.float32),
        "labels": labels,
        "source_detector": sd_zero,
        "edge_index": edge_index,
        "graph_method": graph_method + "+shared_optode",
    }


def load_subject(
    eeg_root: Path,
    fnirs_root: Path,
    subject: int,
    task_key: str,
    epoch_start_s: float = 0.0,
    epoch_stop_s: float = 10.0,
    eeg_epoch_start_s: float | None = None,
    eeg_epoch_stop_s: float | None = None,
) -> SplitArrays:
    """Load 60 strictly paired trials for one subject."""
    import mne

    if epoch_stop_s <= epoch_start_s:
        raise ValueError("epoch_stop_s must be greater than epoch_start_s")
    if eeg_epoch_start_s is None:
        eeg_epoch_start_s = epoch_start_s
    if eeg_epoch_stop_s is None:
        eeg_epoch_stop_s = epoch_stop_s
    if eeg_epoch_stop_s <= eeg_epoch_start_s:
        raise ValueError(
            "eeg_epoch_stop_s must be greater than eeg_epoch_start_s"
        )
    task = TASKS[task_key]
    eeg_start_offset = int(round(eeg_epoch_start_s * 200.0))
    eeg_stop_offset = int(round(eeg_epoch_stop_s * 200.0))
    fnirs_start_offset = int(round(epoch_start_s * 10.0))
    fnirs_stop_offset = int(round(epoch_stop_s * 10.0))
    expected_eeg = eeg_stop_offset - eeg_start_offset
    expected_fnirs = fnirs_stop_offset - fnirs_start_offset

    fnirs_folder = fnirs_root / f"subject {subject:02d}"
    cnt = loadmat(fnirs_folder / "cnt.mat", simplify_cells=True)["cnt"]
    mrk = loadmat(fnirs_folder / "mrk.mat", simplify_cells=True)["mrk"]

    eeg_trials: list[np.ndarray] = []
    fnirs_trials: list[np.ndarray] = []
    labels: list[int] = []
    subjects: list[int] = []
    sessions: list[int] = []
    trial_indices: list[int] = []
    keys: list[str] = []
    details: list[dict] = []
    expected_fnirs_names: list[str] | None = None

    for session_index, session_name in zip(
        task["session_indices"], task["session_names"]
    ):
        eeg_dir = eeg_root / f"sub-{subject}" / session_name / "eeg"
        bdf_path = _exactly_one(eeg_dir, "*_eeg.bdf")
        events_path = _exactly_one(eeg_dir, "*_events.tsv")
        channels_path = _exactly_one(eeg_dir, "*_channels.tsv")
        eeg_names = [
            row["name"] for row in _read_tsv(channels_path)
            if row.get("type", "").upper() == "EEG"
        ]
        if eeg_names != SHIN_EEG_CHANNELS:
            raise RuntimeError(f"Unexpected EEG channel order in {channels_path}")
        raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose="ERROR")
        if abs(float(raw.info["sfreq"]) - 200.0) > 1e-6:
            raise RuntimeError(f"Expected EEG at 200 Hz in {bdf_path}")
        eeg_continuous = np.nan_to_num(
            raw.get_data(picks=SHIN_EEG_CHANNELS, units="uV"),
            copy=False,
        )
        raw.close()
        eeg_events: list[tuple[int, int, str]] = []
        for event in _read_tsv(events_path):
            label_name = event.get("trial_type", "")
            if label_name in task["labels"]:
                eeg_events.append((
                    int(event["sample"]),
                    task["labels"][label_name],
                    label_name,
                ))

        recording, markers = cnt[session_index], mrk[session_index]
        if abs(float(recording["fs"]) - 10.0) > 1e-6:
            raise RuntimeError(f"Expected fNIRS at 10 Hz for subject {subject}")
        fnirs_names = _validate_fnirs_channels(recording["clab"])
        if expected_fnirs_names is None:
            expected_fnirs_names = fnirs_names
        if fnirs_names != expected_fnirs_names:
            raise RuntimeError(f"fNIRS channel order changed for subject {subject}")
        fnirs_continuous = intensity_to_hbo_hbr(
            np.asarray(recording["x"], dtype=np.float64)
        )
        b, a = butter(3, [0.01, 0.1], btype="bandpass", fs=10.0)
        fnirs_continuous = filtfilt(b, a, fnirs_continuous, axis=0)
        times_ms = np.asarray(markers["time"], dtype=np.float64).reshape(-1)
        descriptions = np.asarray(
            markers["event"]["desc"], dtype=np.int64
        ).reshape(-1)
        fnirs_events = [
            (int(round(time_ms * 10.0 / 1000.0)), int(description - 1))
            for time_ms, description in zip(times_ms, descriptions)
        ]
        eeg_sequence = [label for _, label, _ in eeg_events]
        fnirs_sequence = [label for _, label in fnirs_events]
        if (
            len(eeg_events) != 20
            or len(fnirs_events) != 20
            or eeg_sequence != fnirs_sequence
            or Counter(eeg_sequence) != Counter({0: 10, 1: 10})
        ):
            raise RuntimeError(
                f"EEG/fNIRS alignment failed for sub-{subject}, {session_name}: "
                f"EEG={eeg_sequence}, fNIRS={fnirs_sequence}"
            )

        for trial_index, (
            (eeg_sample, label, label_name),
            (fnirs_sample, fnirs_label),
        ) in enumerate(zip(eeg_events, fnirs_events)):
            if label != fnirs_label:
                raise RuntimeError("Label mismatch after sequence validation")
            eeg_start, eeg_stop = (
                eeg_sample + eeg_start_offset,
                eeg_sample + eeg_stop_offset,
            )
            fnirs_start, fnirs_stop = (
                fnirs_sample + fnirs_start_offset,
                fnirs_sample + fnirs_stop_offset,
            )
            if eeg_start < 0 or eeg_stop > eeg_continuous.shape[1]:
                raise RuntimeError(
                    f"EEG epoch out of range: sub-{subject} {session_name} "
                    f"[{eeg_start},{eeg_stop})/{eeg_continuous.shape[1]}"
                )
            if fnirs_sample - 50 < 0 or fnirs_sample - 20 <= fnirs_sample - 50:
                raise RuntimeError(
                    f"Invalid fNIRS baseline: sub-{subject} {session_name} trial {trial_index}"
                )
            if fnirs_start < 0 or fnirs_stop > fnirs_continuous.shape[0]:
                raise RuntimeError(
                    f"fNIRS epoch out of range: sub-{subject} {session_name} "
                    f"[{fnirs_start},{fnirs_stop})/{fnirs_continuous.shape[0]}"
                )
            eeg_trial = eeg_continuous[:, eeg_start:eeg_stop]
            baseline = fnirs_continuous[
                fnirs_sample - 50:fnirs_sample - 20
            ].mean(axis=0, keepdims=True)
            fnirs_trial = fnirs_continuous[fnirs_start:fnirs_stop] - baseline
            if eeg_trial.shape != (30, expected_eeg):
                raise RuntimeError(f"Bad EEG trial shape {eeg_trial.shape}")
            if fnirs_trial.shape != (expected_fnirs, 36, 2):
                raise RuntimeError(f"Bad fNIRS trial shape {fnirs_trial.shape}")
            if not np.isfinite(eeg_trial).all() or not np.isfinite(fnirs_trial).all():
                raise RuntimeError(
                    f"NaN/Inf in sub-{subject} {session_name} trial {trial_index}"
                )
            key = (
                f"sub-{subject:02d}/{task_key}/{session_name}/"
                f"trial-{trial_index:02d}/start-{epoch_start_s:g}s"
            )
            eeg_trials.append(preprocess_eeg_cbramod(eeg_trial))
            fnirs_trials.append(
                fnirs_trial.transpose(1, 2, 0).astype(np.float32)
            )
            labels.append(label)
            subjects.append(subject)
            sessions.append(session_index)
            trial_indices.append(trial_index)
            keys.append(key)
        details.append({
            "subject": subject,
            "session": session_name,
            "session_index": session_index,
            "trials": 20,
            "label_counts": dict(Counter(eeg_sequence)),
            "event_sequence_aligned": True,
            "eeg_first_sample": int(eeg_events[0][0]),
            "fnirs_first_sample": int(fnirs_events[0][0]),
        })

    arrays = SplitArrays(
        eeg_uv=np.stack(eeg_trials),
        fnirs_um=np.stack(fnirs_trials),
        labels=np.asarray(labels, dtype=np.int64),
        subjects=np.asarray(subjects, dtype=np.int16),
        sessions=np.asarray(sessions, dtype=np.int8),
        trials=np.asarray(trial_indices, dtype=np.int8),
        keys=np.asarray(keys, dtype=np.str_),
        details=details,
    )
    if (
        arrays.eeg_uv.shape != (60, 30, expected_eeg)
        or arrays.fnirs_um.shape != (60, 36, 2, expected_fnirs)
        or Counter(arrays.labels.tolist()) != Counter({0: 30, 1: 30})
        or len(set(arrays.keys.tolist())) != 60
    ):
        raise RuntimeError(
            f"Unexpected subject output: EEG={arrays.eeg_uv.shape}, "
            f"fNIRS={arrays.fnirs_um.shape}, labels={Counter(arrays.labels.tolist())}"
        )
    return arrays


def load_split(
    eeg_root: Path,
    fnirs_root: Path,
    subjects: list[int],
    split_name: str,
    task_key: str,
    cache_dir: Path | None,
    epoch_start_s: float = 0.0,
    epoch_stop_s: float = 10.0,
    eeg_epoch_start_s: float | None = None,
    eeg_epoch_stop_s: float | None = None,
) -> SplitArrays:
    if eeg_epoch_start_s is None:
        eeg_epoch_start_s = epoch_start_s
    if eeg_epoch_stop_s is None:
        eeg_epoch_stop_s = epoch_stop_s
    epoch_tag = (
        f"{epoch_start_s:g}_{epoch_stop_s:g}".replace("-", "m").replace(".", "p")
    )
    eeg_epoch_tag = (
        f"{eeg_epoch_start_s:g}_{eeg_epoch_stop_s:g}"
        .replace("-", "m")
        .replace(".", "p")
    )
    if eeg_epoch_tag != epoch_tag:
        epoch_tag = f"fnirs-{epoch_tag}_eeg-{eeg_epoch_tag}"
    tag = "-".join(map(str, subjects))
    cache_path = (
        cache_dir
        / f"{task_key}_{split_name}_sub-{tag}_paired_hbo-hbr_epoch-{epoch_tag}.npz"
        if cache_dir is not None
        else None
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.is_file():
            item = np.load(cache_path, allow_pickle=False)
            details_path = cache_path.with_suffix(".details.json")
            details = (
                json.loads(details_path.read_text(encoding="utf-8"))
                if details_path.is_file()
                else [{"source": "cache", "subjects": subjects}]
            )
            return SplitArrays(
                eeg_uv=item["eeg_uv"],
                fnirs_um=item["fnirs_um"],
                labels=item["labels"],
                subjects=item["subjects"],
                sessions=item["sessions"],
                trials=item["trials"],
                keys=item["keys"],
                details=details,
            )

    parts = [
        load_subject(
            eeg_root, fnirs_root, subject, task_key,
            epoch_start_s, epoch_stop_s,
            eeg_epoch_start_s, eeg_epoch_stop_s,
        )
        for subject in subjects
    ]
    result = SplitArrays(
        eeg_uv=np.concatenate([part.eeg_uv for part in parts]),
        fnirs_um=np.concatenate([part.fnirs_um for part in parts]),
        labels=np.concatenate([part.labels for part in parts]),
        subjects=np.concatenate([part.subjects for part in parts]),
        sessions=np.concatenate([part.sessions for part in parts]),
        trials=np.concatenate([part.trials for part in parts]),
        keys=np.concatenate([part.keys for part in parts]),
        details=[detail for part in parts for detail in part.details],
    )
    if len(set(result.keys.tolist())) != len(result.keys):
        raise RuntimeError(f"Duplicate trial keys in split {split_name}")
    if cache_path is not None:
        np.savez_compressed(
            cache_path,
            eeg_uv=result.eeg_uv,
            fnirs_um=result.fnirs_um,
            labels=result.labels,
            subjects=result.subjects,
            sessions=result.sessions,
            trials=result.trials,
            keys=result.keys,
        )
        cache_path.with_suffix(".details.json").write_text(
            json.dumps(result.details, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return result


def fit_train_fnirs_stats(fnirs_um: np.ndarray) -> dict[str, np.ndarray]:
    """Fit node- and chromophore-specific statistics on the training split."""
    if fnirs_um.ndim != 4 or fnirs_um.shape[1:3] != (36, 2):
        raise ValueError(f"Expected [trial,36,2,time], got {fnirs_um.shape}")
    mean = fnirs_um.mean(axis=(0, 3), keepdims=True, dtype=np.float64)
    std = fnirs_um.std(axis=(0, 3), keepdims=True, dtype=np.float64)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Non-finite fNIRS normalization statistics")
    std = np.maximum(std, 1e-6)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def normalize_fnirs(
    fnirs_um: np.ndarray,
    stats: dict[str, np.ndarray],
) -> np.ndarray:
    value = (fnirs_um - stats["mean"]) / stats["std"]
    if not np.isfinite(value).all():
        raise ValueError("Non-finite normalized fNIRS values")
    return value.astype(np.float32, copy=False)


class SHINTrialDataset(Dataset):
    """Trial-level paired samples; no window-level label multiplication."""

    MODALITY_INDICES = {
        "hbo": (0,),
        "hbr": (1,),
        "hbo_hbr": (0, 1),
    }

    def __init__(
        self,
        arrays: SplitArrays,
        fnirs_stats: dict[str, np.ndarray],
        fnirs_modalities: str,
    ) -> None:
        if fnirs_modalities not in self.MODALITY_INDICES:
            raise ValueError(f"Unknown fNIRS modalities: {fnirs_modalities}")
        indices = self.MODALITY_INDICES[fnirs_modalities]
        self.eeg_uv = arrays.eeg_uv
        self.fnirs = normalize_fnirs(arrays.fnirs_um, fnirs_stats)[:, :, indices, :]
        self.labels = arrays.labels
        self.subjects = arrays.subjects
        self.keys = arrays.keys

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.eeg_uv[index]),
            torch.from_numpy(self.fnirs[index]),
            torch.tensor(self.labels[index], dtype=torch.long),
            torch.tensor(index, dtype=torch.long),
        )
