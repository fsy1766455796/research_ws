"""Replace the frontend yaw of every pair by the integrated D400 gyroscope yaw.

Translation stays from the visual (monocular DA3-BEV) frontend.  The gyro
bias is estimated per sequence without ground truth (``--bias visual_still``:
median gyro rate over pairs the visual frontend reports as non-turning) or
not corrected (``--bias none``).  Output has the same CSV schema, so all
evaluation, SLAM and plotting tools work unchanged.
"""

import argparse
import csv
import json
from pathlib import Path

from .gyro_yaw import GyroYaw
from .openloris_data import resolve_sequence_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bias', choices=('none', 'visual_still'), default='visual_still')
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error('output exists')
    with open(args.predictions, encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    biases = {}
    for sequence in sorted({r['sequence'] for r in rows}):
        gyro = GyroYaw(resolve_sequence_dir(args.dataset_root, sequence))
        own = [r for r in rows if r['sequence'] == sequence]
        if args.bias == 'visual_still':
            gyro.estimate_bias([(float(r['timestamp_0']), float(r['timestamp_1'])) for r in own],
                               [float(r['pred_yaw_rad']) for r in own])
        biases[sequence] = gyro.bias
        for r in own:
            r['pred_yaw_rad'] = repr(gyro.increment(float(r['timestamp_0']), float(r['timestamp_1'])))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'bias_rad_per_s': biases, 'mode': args.bias}, indent=1))


if __name__ == '__main__':
    main()
