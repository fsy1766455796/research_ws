"""Command-line preprocessing for OpenLORIS learned BEV odometry."""

import argparse
import json
from pathlib import Path

from .multichannel_bev import MultiChannelBEVConfig
from .openloris_data import (
    build_pair_records,
    prepare_sequence_bevs,
    save_pair_manifest,
)


DEFAULT_TRAIN = ("office1-1", "office1-2", "office1-3", "office1-4")
DEFAULT_VALIDATION = ("office1-5",)
DEFAULT_TEST = ("office1-6", "office1-7")


def _comma_list(value):
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_argument_parser():
    """Build the preprocessing command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Precompute multi-channel BEVs and supervised pair manifests."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument(
        "--depth-source", choices=("aligned_depth", "da3"), default="da3"
    )
    parser.add_argument("--da3-root")
    parser.add_argument(
        "--train-sequences", type=_comma_list,
        default=DEFAULT_TRAIN,
    )
    parser.add_argument(
        "--validation-sequences", type=_comma_list,
        default=DEFAULT_VALIDATION,
    )
    parser.add_argument(
        "--test-sequences", type=_comma_list,
        default=DEFAULT_TEST,
    )
    parser.add_argument("--train-strides", type=_comma_list, default=("3", "6", "9", "12"))
    parser.add_argument("--eval-stride", type=int, default=6)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--resolution-m", type=float, default=0.05)
    parser.add_argument("--forward-min-m", type=float, default=-1.0)
    parser.add_argument("--forward-max-m", type=float, default=6.0)
    parser.add_argument("--left-min-m", type=float, default=-3.5)
    parser.add_argument("--left-max-m", type=float, default=3.5)
    parser.add_argument("--height-min-m", type=float, default=-0.2)
    parser.add_argument("--height-max-m", type=float, default=1.8)
    parser.add_argument("--max-translation-m", type=float, default=0.8)
    parser.add_argument("--max-yaw-deg", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    """Precompute BEVs and write train/validation/test manifests."""
    args = build_argument_parser().parse_args(argv)
    if args.depth_source == "da3" and not args.da3_root:
        raise SystemExit("--da3-root is required for --depth-source da3")
    config = MultiChannelBEVConfig(
        forward_min_m=args.forward_min_m,
        forward_max_m=args.forward_max_m,
        left_min_m=args.left_min_m,
        left_max_m=args.left_max_m,
        height_min_m=args.height_min_m,
        height_max_m=args.height_max_m,
        resolution_m=args.resolution_m,
    )
    all_sequences = tuple(dict.fromkeys(
        args.train_sequences + args.validation_sequences + args.test_sequences
    ))
    summaries = []
    for sequence in all_sequences:
        summary = prepare_sequence_bevs(
            dataset_root=args.dataset_root,
            cache_root=args.cache_root,
            sequence=sequence,
            depth_source=args.depth_source,
            da3_root=args.da3_root,
            config=config,
            pixel_stride=args.pixel_stride,
            overwrite=args.overwrite,
        )
        summaries.append(summary)
        print(json.dumps(summary))

    manifest_dir = Path(args.cache_root) / "manifests"
    splits = {
        "train": (args.train_sequences, tuple(int(v) for v in args.train_strides)),
        "validation": (args.validation_sequences, (args.eval_stride,)),
        "test": (args.test_sequences, (args.eval_stride,)),
    }
    counts = {}
    for split, (sequences, strides) in splits.items():
        records = build_pair_records(
            dataset_root=args.dataset_root,
            cache_root=args.cache_root,
            sequences=sequences,
            strides=strides,
            max_translation_m=args.max_translation_m,
            max_yaw_deg=args.max_yaw_deg,
        )
        save_pair_manifest(records, manifest_dir / f"{split}.jsonl")
        counts[split] = len(records)
    run_metadata = {
        "depth_source": args.depth_source,
        "dataset_root": str(Path(args.dataset_root).expanduser()),
        "cache_root": str(Path(args.cache_root).expanduser()),
        "splits": counts,
        "train_sequences": list(args.train_sequences),
        "validation_sequences": list(args.validation_sequences),
        "test_sequences": list(args.test_sequences),
        "train_strides": [int(v) for v in args.train_strides],
        "eval_stride": args.eval_stride,
        "bev_config": config.__dict__,
    }
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with (manifest_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(run_metadata, stream, indent=2)
    print(json.dumps(run_metadata, indent=2))


if __name__ == "__main__":
    main()
