"""Replay existing backend; ground truth is used only after each match to score it."""

import argparse
import csv
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import local_map_backend as backend
from .se2 import wrap_angle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    with Path(args.predictions).open(encoding='utf-8-sig', newline='') as stream:
        predictions = list(csv.DictReader(stream))
    with Path(args.manifest).open(encoding='utf-8') as stream:
        manifest = {backend.pair_key(row): row for row in map(json.loads, filter(str.strip, stream))}
    cfg = backend.BackendConfig()
    original_match = backend.match_map
    original_optimize = backend.optimize_window
    all_rows, reports = [], []
    for segment, (sequence, chain) in enumerate(backend.select_chains(predictions)):
        records = [manifest[backend.pair_key(row)] for row in chain]
        filenames = [records[0]['bev_0']] + [r['bev_1'] for r in records]
        increments = np.array([[float(row[k]) for k in
                               ('pred_forward_m', 'pred_left_m', 'pred_yaw_rad')] for row in chain])
        truth = [np.zeros(3)]
        for row in records:
            truth.append(backend.compose(truth[-1], [row[k] for k in ('forward_m', 'left_m', 'yaw_rad')]))
        calls, active_counts, rows = [1], [], []

        def measured_match(source, target, initial, config):
            measured, diagnostic = original_match(source, target, initial, config)
            calls[0] += 1
            index = calls[0]
            start = max(0, index-config.window)
            # Scoring happens after matching; no true pose enters the matcher.
            reference = backend.relative(truth[start], truth[index])
            before = math.degrees(wrap_angle(initial[2]-reference[2]))
            after = math.degrees(wrap_angle(measured[2]-reference[2]))
            row = dict(sequence=sequence, segment=segment, index=index,
                       accepted=diagnostic['accepted'], before_yaw_error_deg=before,
                       after_yaw_error_deg=after,
                       signed_yaw_correction_deg=math.degrees(wrap_angle(measured[2]-initial[2])),
                       yaw_improved=abs(after)<abs(before),
                       **{k: diagnostic.get(k) for k in ['overlap', 'rmse_m', 'reason']})
            rows.append(row)
            return measured, diagnostic

        def measured_optimize(poses, increments, edges, start, config):
            active_counts.append(sum(i >= start for i, _, _ in edges))
            return original_optimize(poses, increments, edges, start, config)

        with patch.object(backend, 'match_map', measured_match), patch.object(backend, 'optimize_window', measured_optimize):
            raw, corrected, _ = backend.run_backend(increments, lambda i: backend.load_points(filenames[i], cfg), cfg)
        accepted = [row for row in rows if row['accepted']]
        reports.append(dict(sequence=sequence, segment=segment, accepted=len(accepted),
             accepted_yaw_improved_fraction=float(np.mean([r['yaw_improved'] for r in accepted])) if accepted else None,
             accepted_before_local_yaw_mae_deg=float(np.mean([abs(r['before_yaw_error_deg']) for r in accepted])) if accepted else None,
             accepted_after_local_yaw_mae_deg=float(np.mean([abs(r['after_yaw_error_deg']) for r in accepted])) if accepted else None,
             mean_abs_yaw_correction_deg=float(np.mean([abs(r['signed_yaw_correction_deg']) for r in accepted])) if accepted else None,
             active_map_edges_last_20=active_counts[-20:],
             raw=backend.metrics(raw, np.asarray(truth)), optimized=backend.metrics(corrected, np.asarray(truth))))
        all_rows.extend(rows)
    backend.write_csv(output / 'constraints.csv', all_rows)
    (output / 'summary.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
