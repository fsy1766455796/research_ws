"""OpenLORIS-Scene preprocessing and pair indexing for learned BEV odometry."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from .multichannel_bev import (
    MultiChannelBEVConfig,
    OPENLORIS_COLOR_INTRINSICS,
    OPENLORIS_COLOR_SIZE,
    OPENLORIS_T_BASE_COLOR,
    depth_to_multichannel_bev,
    scale_intrinsics,
)
from .openloris_geometry import (
    interpolate_pose,
    load_groundtruth,
    load_timestamped_paths,
    relative_planar_pose,
)


@dataclass(frozen=True)
class PairRecord:
    """Describe one cached BEV pair and its relative base pose."""

    sequence: str
    timestamp_0: float
    timestamp_1: float
    bev_0: str
    bev_1: str
    forward_m: float
    left_m: float
    yaw_rad: float

    @property
    def target(self):
        """Return the pose label as a float32 NumPy array."""
        return np.array(
            [self.forward_m, self.left_m, self.yaw_rad], dtype=np.float32
        )


def resolve_sequence_dir(dataset_root, sequence):
    """Locate one extracted sequence while tolerating one extra scene folder."""
    root = Path(dataset_root).expanduser().resolve()
    direct = root / sequence
    if (direct / "color.txt").is_file():
        return direct
    matches = [
        path for path in root.glob(f"**/{sequence}")
        if (path / "color.txt").is_file()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one extracted directory for {sequence} under {root}; "
            f"found {len(matches)}"
        )
    return matches[0]


def bev_cache_path(cache_root, sequence, timestamp):
    """Return the canonical cache filename for one timestamp."""
    return Path(cache_root) / sequence / f"{float(timestamp):.8f}.npz"


def _nearest_entry(entries, timestamp, max_difference_sec):
    times = np.asarray([entry.timestamp for entry in entries], dtype=np.float64)
    index = int(np.searchsorted(times, timestamp))
    candidates = [candidate for candidate in (index - 1, index)
                  if 0 <= candidate < len(entries)]
    if not candidates:
        return None
    best = min(candidates, key=lambda candidate: abs(times[candidate] - timestamp))
    if abs(times[best] - timestamp) > max_difference_sec:
        return None
    return entries[best]


def _load_sensor_depth(filename):
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required to read OpenLORIS PNG depth") from error
    with Image.open(filename) as image:
        depth_mm = np.asarray(image, dtype=np.float32)
    return depth_mm * 0.001


def _resolve_da3_prediction(da3_root, sequence, color_relative_path, timestamp):
    root = Path(da3_root)
    relative = Path(color_relative_path)
    names = (
        root / sequence / relative.with_suffix(".npy"),
        root / sequence / f"{relative.stem}.npy",
        root / sequence / relative.with_suffix(".npz"),
        root / sequence / f"{relative.stem}.npz",
        root / f"{float(timestamp):.8f}.npy",
        root / f"{float(timestamp):.8f}.npz",
    )
    for candidate in names:
        if candidate.is_file():
            return candidate
    return None


def _load_da3_depth(filename):
    loaded = np.load(filename, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            key = "depth" if "depth" in loaded.files else loaded.files[0]
            depth = loaded[key]
        finally:
            loaded.close()
    else:
        depth = loaded
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"DA3 depth must be [H,W], got {depth.shape}: {filename}")
    return depth


def prepare_sequence_bevs(
    dataset_root,
    cache_root,
    sequence,
    depth_source="aligned_depth",
    da3_root=None,
    config=None,
    pixel_stride=2,
    overwrite=False,
):
    """Precompute compressed multi-channel BEVs for one sequence."""
    cfg = config or MultiChannelBEVConfig()
    sequence_dir = resolve_sequence_dir(dataset_root, sequence)
    colors = load_timestamped_paths(sequence_dir / "color.txt")
    aligned_depths = None
    if depth_source == "aligned_depth":
        aligned_depths = load_timestamped_paths(
            sequence_dir / "aligned_depth.txt"
        )
    elif depth_source == "da3" and da3_root is None:
        raise ValueError("da3_root is required when depth_source='da3'")
    elif depth_source != "da3":
        raise ValueError("depth_source must be 'aligned_depth' or 'da3'")

    output_dir = Path(cache_root) / sequence
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    missing = 0
    for color in colors:
        output = bev_cache_path(cache_root, sequence, color.timestamp)
        if output.is_file() and not overwrite:
            continue
        if depth_source == "aligned_depth":
            depth_entry = _nearest_entry(
                aligned_depths, color.timestamp, max_difference_sec=0.04
            )
            if depth_entry is None:
                missing += 1
                continue
            depth = _load_sensor_depth(sequence_dir / depth_entry.path)
        else:
            prediction = _resolve_da3_prediction(
                da3_root, sequence, color.path, color.timestamp
            )
            if prediction is None:
                missing += 1
                continue
            depth = _load_da3_depth(prediction)
        intrinsics = scale_intrinsics(
            OPENLORIS_COLOR_INTRINSICS,
            OPENLORIS_COLOR_SIZE,
            (depth.shape[1], depth.shape[0]),
        )
        bev = depth_to_multichannel_bev(
            depth,
            config=cfg,
            intrinsics=intrinsics,
            t_base_camera=OPENLORIS_T_BASE_COLOR,
            pixel_stride=pixel_stride,
        )
        np.savez_compressed(output, bev=bev.astype(np.float16))
        written += 1

    metadata = {
        "sequence": sequence,
        "depth_source": depth_source,
        "frames_in_color_index": len(colors),
        "frames_written_this_run": written,
        "frames_missing": missing,
        "channels": list(cfg.channel_names),
        "bev_config": asdict(cfg),
        "intrinsics_fx_fy_cx_cy": list(OPENLORIS_COLOR_INTRINSICS),
        "coordinate_convention": "forward_left_up_metres",
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return metadata


def build_pair_records(
    dataset_root,
    cache_root,
    sequences,
    strides=(3, 6, 9, 12),
    max_translation_m=0.8,
    max_yaw_deg=20.0,
    max_groundtruth_gap_sec=0.05,
):
    """Build supervised frame pairs whose target lies inside matcher bounds."""
    records = []
    for sequence in sequences:
        sequence_dir = resolve_sequence_dir(dataset_root, sequence)
        colors = load_timestamped_paths(sequence_dir / "color.txt")
        groundtruth = load_groundtruth(sequence_dir / "groundtruth.txt")
        poses = {
            entry.timestamp: interpolate_pose(
                groundtruth, entry.timestamp, max_gap_sec=max_groundtruth_gap_sec
            )
            for entry in colors
        }
        for first_index, first in enumerate(colors):
            pose_0 = poses[first.timestamp]
            bev_0 = bev_cache_path(cache_root, sequence, first.timestamp)
            if pose_0 is None or not bev_0.is_file():
                continue
            for stride in strides:
                second_index = first_index + int(stride)
                if second_index >= len(colors):
                    continue
                second = colors[second_index]
                pose_1 = poses[second.timestamp]
                bev_1 = bev_cache_path(cache_root, sequence, second.timestamp)
                if pose_1 is None or not bev_1.is_file():
                    continue
                target = relative_planar_pose(pose_0, pose_1)
                if np.hypot(target[0], target[1]) > max_translation_m:
                    continue
                if abs(math.degrees(float(target[2]))) > max_yaw_deg:
                    continue
                records.append(
                    PairRecord(
                        sequence=sequence,
                        timestamp_0=first.timestamp,
                        timestamp_1=second.timestamp,
                        bev_0=str(bev_0),
                        bev_1=str(bev_1),
                        forward_m=float(target[0]),
                        left_m=float(target[1]),
                        yaw_rad=float(target[2]),
                    )
                )
    return records


def save_pair_manifest(records, filename):
    """Write pair records as newline-delimited JSON."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record)) + "\n")


def load_pair_manifest(filename):
    """Read pair records from newline-delimited JSON."""
    records = []
    with Path(filename).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(PairRecord(**json.loads(line)))
    return records
