"""Precompute local-map BEV pairs for learned odometry fine-tuning."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from .learned_bev_model import warp_current_to_previous
from .openloris_data import (
    PairRecord,
    bev_cache_path,
    load_pair_manifest,
    resolve_sequence_dir,
    save_pair_manifest,
)
from .openloris_geometry import (
    interpolate_pose,
    load_groundtruth,
    load_timestamped_paths,
    relative_planar_pose,
)


DEFAULT_SPLITS = ("train", "validation", "test")
DEFAULT_METRIC_BOUNDS = (-1.0, 6.0, -3.5, 3.5)


def _device_from_argument(value):
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_bev(filename):
    with np.load(filename, allow_pickle=False) as archive:
        return np.asarray(archive["bev"], dtype=np.float32)


def _local_map_path(output_root, sequence, timestamp):
    return Path(output_root) / "local_maps" / sequence / f"{float(timestamp):.8f}.npz"


def _build_sequence_context(dataset_root, cache_root, sequence):
    sequence_dir = resolve_sequence_dir(dataset_root, sequence)
    colors = load_timestamped_paths(sequence_dir / "color.txt")
    groundtruth = load_groundtruth(sequence_dir / "groundtruth.txt")
    poses = {}
    valid_entries = []
    for entry in colors:
        pose = interpolate_pose(groundtruth, entry.timestamp)
        bev = bev_cache_path(cache_root, sequence, entry.timestamp)
        if pose is not None and bev.is_file():
            poses[entry.timestamp] = pose
            valid_entries.append(entry)
    index_by_timestamp = {
        round(entry.timestamp, 7): index for index, entry in enumerate(valid_entries)
    }
    return valid_entries, poses, index_by_timestamp


def _fuse_local_map(
    entries,
    poses,
    cache_root,
    sequence,
    reference_index,
    window_size,
    metric_bounds,
    device,
):
    reference = entries[reference_index]
    reference_pose = poses[reference.timestamp]
    start = max(0, reference_index - int(window_size) + 1)
    warped = []
    for entry in entries[start:reference_index + 1]:
        source_bev = _load_bev(bev_cache_path(cache_root, sequence, entry.timestamp))
        source_in_reference = relative_planar_pose(
            reference_pose, poses[entry.timestamp]
        )
        source = torch.from_numpy(source_bev).unsqueeze(0).to(device)
        pose = torch.from_numpy(source_in_reference).unsqueeze(0).to(device)
        warped.append(
            warp_current_to_previous(
                source, pose, metric_bounds
            )[0].detach().cpu()
        )
    return torch.amax(torch.stack(warped, dim=0), dim=0).numpy()


def _prepare_split(
    split,
    manifest_path,
    dataset_root,
    cache_root,
    output_root,
    window_size,
    metric_bounds,
    device,
    overwrite,
):
    records = load_pair_manifest(manifest_path)
    contexts = {}
    output_records = []
    written = 0
    reused = 0
    skipped = 0
    for record in records:
        if record.sequence not in contexts:
            contexts[record.sequence] = _build_sequence_context(
                dataset_root, cache_root, record.sequence
            )
        entries, poses, index_by_timestamp = contexts[record.sequence]
        timestamp_key = round(record.timestamp_0, 7)
        if timestamp_key not in index_by_timestamp:
            skipped += 1
            continue
        output = _local_map_path(output_root, record.sequence, record.timestamp_0)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.is_file() and not overwrite:
            reused += 1
        else:
            local_map = _fuse_local_map(
                entries,
                poses,
                cache_root,
                record.sequence,
                index_by_timestamp[timestamp_key],
                window_size,
                metric_bounds,
                device,
            )
            np.savez_compressed(output, bev=local_map.astype(np.float16))
            written += 1
        output_records.append(
            PairRecord(
                sequence=record.sequence,
                timestamp_0=record.timestamp_0,
                timestamp_1=record.timestamp_1,
                bev_0=str(output),
                bev_1=record.bev_1,
                forward_m=record.forward_m,
                left_m=record.left_m,
                yaw_rad=record.yaw_rad,
            )
        )
    output_manifest = Path(output_root) / "manifests" / f"{split}.jsonl"
    save_pair_manifest(output_records, output_manifest)
    return {
        "split": split,
        "source_manifest": str(Path(manifest_path).resolve()),
        "output_manifest": str(output_manifest.resolve()),
        "records_in": len(records),
        "records_out": len(output_records),
        "local_maps_written": written,
        "local_maps_reused": reused,
        "records_skipped": skipped,
    }


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-manifest-dir")
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--forward-min-m", type=float, default=-1.0)
    parser.add_argument("--forward-max-m", type=float, default=6.0)
    parser.add_argument("--left-min-m", type=float, default=-3.5)
    parser.add_argument("--left-max-m", type=float, default=3.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    source_manifest_dir = (
        Path(args.source_manifest_dir)
        if args.source_manifest_dir
        else Path(args.cache_root) / "manifests"
    )
    splits = tuple(item.strip() for item in args.splits.split(",") if item.strip())
    device = _device_from_argument(args.device)
    metric_bounds = (
        args.forward_min_m,
        args.forward_max_m,
        args.left_min_m,
        args.left_max_m,
    )
    summaries = []
    for split in splits:
        manifest = source_manifest_dir / f"{split}.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing source manifest: {manifest}")
        summary = _prepare_split(
            split=split,
            manifest_path=manifest,
            dataset_root=args.dataset_root,
            cache_root=args.cache_root,
            output_root=args.output_root,
            window_size=args.window_size,
            metric_bounds=metric_bounds,
            device=device,
            overwrite=args.overwrite,
        )
        summaries.append(summary)
        print(json.dumps(summary))
    metadata = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "cache_root": str(Path(args.cache_root).resolve()),
        "output_root": str(Path(args.output_root).resolve()),
        "source_manifest_dir": str(source_manifest_dir.resolve()),
        "window_size": args.window_size,
        "metric_bounds": metric_bounds,
        "splits": summaries,
        "format": (
            "bev_0 is a ground-truth fused local map at timestamp_0; "
            "bev_1 is the original current-frame BEV."
        ),
    }
    manifest_dir = Path(args.output_root) / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with (manifest_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
