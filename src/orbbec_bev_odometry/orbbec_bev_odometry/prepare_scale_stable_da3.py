"""Build a DA3 BEV cache with temporally stabilised depth scale (monocular only).

Monocular DA3 depth has frame-to-frame global scale jitter (~2.5-3 % std of
log median depth between consecutive frames on office1, versus 0.5-1.5 % for
the sensor depth), which shows up as translation noise in the BEV frontend.
Each frame's depth is multiplied by ``S_t / s_t`` where ``s_t`` is its median
valid depth and ``S_t`` the median of ``s`` over +-``window`` neighbouring
frames (factor clipped to [1-clip, 1+clip]).  Only DA3 itself is used; no
sensor depth and no ground truth.  The cache layout and pair manifests mirror
an existing DA3 cache so that every downstream tool works unchanged.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .multichannel_bev import (OPENLORIS_COLOR_INTRINSICS, OPENLORIS_COLOR_SIZE, OPENLORIS_T_BASE_COLOR,
                               MultiChannelBEVConfig, depth_to_multichannel_bev, scale_intrinsics)
from .openloris_data import _load_da3_depth, _resolve_da3_prediction, bev_cache_path, resolve_sequence_dir
from .openloris_geometry import load_timestamped_paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--da3-root', required=True)
    parser.add_argument('--reference-cache', required=True, help='existing DA3 cache whose manifests are mirrored')
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--sequences', default='office1-1,office1-2,office1-3,office1-4,office1-5,office1-6,office1-7')
    parser.add_argument('--window', type=int, default=15)
    parser.add_argument('--clip', type=float, default=0.15)
    parser.add_argument('--pixel-stride', type=int, default=2)
    args = parser.parse_args(argv)
    cache = Path(args.cache_root)
    if cache.exists():
        parser.error('cache root exists')
    cfg = MultiChannelBEVConfig()
    report = {}
    for sequence in args.sequences.split(','):
        seq_dir = resolve_sequence_dir(args.dataset_root, sequence)
        colors = load_timestamped_paths(seq_dir / 'color.txt')
        files = [_resolve_da3_prediction(args.da3_root, sequence, c.path, c.timestamp) for c in colors]
        medians = np.full(len(colors), np.nan)
        for i, f in enumerate(files):
            if f is not None:
                d = _load_da3_depth(f)
                v = d[::4, ::4]
                v = v[np.isfinite(v) & (v > 0.3) & (v < 10)]
                if v.size:
                    medians[i] = np.median(v)
        factors = np.ones(len(colors))
        for i in range(len(colors)):
            if not np.isfinite(medians[i]):
                continue
            win = medians[max(0, i - args.window):i + args.window + 1]
            win = win[np.isfinite(win)]
            factors[i] = np.clip(np.median(win) / medians[i], 1 - args.clip, 1 + args.clip)
        out_dir = cache / sequence
        out_dir.mkdir(parents=True)
        written = 0
        for c, f, k in zip(colors, files, factors):
            if f is None:
                continue
            depth = _load_da3_depth(f) * k
            intr = scale_intrinsics(OPENLORIS_COLOR_INTRINSICS, OPENLORIS_COLOR_SIZE, (depth.shape[1], depth.shape[0]))
            bev = depth_to_multichannel_bev(depth, config=cfg, intrinsics=intr, t_base_camera=OPENLORIS_T_BASE_COLOR,
                                            pixel_stride=args.pixel_stride)
            np.savez_compressed(bev_cache_path(cache, sequence, c.timestamp), bev=bev.astype(np.float16))
            written += 1
        meta = dict(sequence=sequence, depth_source='da3_scale_stable', frames_written=written,
                    window=args.window, clip=args.clip, factor_std=float(np.std(factors)),
                    channels=list(cfg.channel_names), bev_config=asdict(cfg),
                    coordinate_convention='forward_left_up_metres')
        (out_dir / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        report[sequence] = dict(written=written, factor_std=float(np.std(factors)))
        print(sequence, report[sequence], flush=True)
    # mirror manifests
    ref = Path(args.reference_cache)
    (cache / 'manifests').mkdir()
    for manifest in (ref / 'manifests').glob('*.jsonl'):
        text = manifest.read_text(encoding='utf-8')
        for old in (str(ref), str(ref).replace('\\', '\\\\'), str(ref).replace('\\', '/')):
            new = str(cache) if old == str(ref) else (str(cache).replace('\\', '\\\\') if '\\\\' in old else str(cache).replace('\\', '/'))
            text = text.replace(old, new)
        (cache / 'manifests' / manifest.name).write_text(text, encoding='utf-8')
    (cache / 'manifests' / 'scale_stable_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
