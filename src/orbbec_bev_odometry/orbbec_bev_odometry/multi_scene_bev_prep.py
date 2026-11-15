"""BEV cache + pair manifest for the SLAM/map stage on any OpenLORIS sequence.

Reads a ``multi_scene_pnp`` predictions.csv, takes the frames of the chains the
SLAM backend will use (``select_chains``), writes a DA3 BEV for each of them
with the sequence's own calibration (``sequence_calibration``), and writes a
manifest (jsonl) linking every pair to its BEV files.
"""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from .local_map_backend import select_chains
from .multichannel_bev import MultiChannelBEVConfig, depth_to_multichannel_bev, scale_intrinsics
from .openloris_data import (_load_da3_depth, _resolve_da3_prediction, bev_cache_path, resolve_sequence_dir,
                             sequence_calibration)
from .openloris_geometry import load_timestamped_paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--da3-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args(argv)
    if Path(args.manifest).exists():
        parser.error('manifest exists')
    with open(args.predictions, encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    cfg = MultiChannelBEVConfig()
    needed = {}
    for sequence, chain in select_chains(rows):
        for r in chain:
            needed.setdefault(sequence, set()).update((round(float(r['timestamp_0']), 6), round(float(r['timestamp_1']), 6)))
    t0 = time.time()
    for sequence, stamps in needed.items():
        d = resolve_sequence_dir(args.dataset_root, sequence)
        intr, size, t_bc = sequence_calibration(args.dataset_root, sequence)
        colors = {round(c.timestamp, 6): c for c in load_timestamped_paths(d / 'color.txt')}
        out_dir = Path(args.cache_root) / sequence
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for n, ts in enumerate(sorted(stamps)):
            c = colors[ts]
            out = bev_cache_path(args.cache_root, sequence, c.timestamp)
            if out.is_file():
                continue
            f = _resolve_da3_prediction(args.da3_root, sequence, c.path, c.timestamp)
            if f is None:
                continue
            depth = _load_da3_depth(f)
            bev = depth_to_multichannel_bev(depth, config=cfg,
                                            intrinsics=scale_intrinsics(intr, size, (depth.shape[1], depth.shape[0])),
                                            t_base_camera=t_bc, pixel_stride=2)
            np.savez_compressed(out, bev=bev.astype(np.float16))
            written += 1
            if n % 1000 == 0:
                print(sequence, n, len(stamps), f'{time.time() - t0:.0f}s', flush=True)
        (out_dir / 'metadata.json').write_text(json.dumps(dict(
            sequence=sequence, depth_source='da3', frames_written_this_run=written, bev_config=asdict(cfg),
            intrinsics_fx_fy_cx_cy=list(intr), t_base_color=t_bc.tolist(),
            coordinate_convention='forward_left_up_metres'), indent=2), encoding='utf-8')
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, 'w', encoding='utf-8') as stream:
        for r in rows:
            seq = r['sequence']
            stream.write(json.dumps(dict(
                sequence=seq, timestamp_0=float(r['timestamp_0']), timestamp_1=float(r['timestamp_1']),
                bev_0=str(bev_cache_path(args.cache_root, seq, float(r['timestamp_0']))),
                bev_1=str(bev_cache_path(args.cache_root, seq, float(r['timestamp_1']))),
                forward_m=float(r['target_forward_m']), left_m=float(r['target_left_m']),
                yaw_rad=float(r['target_yaw_rad']))) + '\n')


if __name__ == '__main__':
    main()
