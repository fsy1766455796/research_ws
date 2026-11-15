"""Stitch BEV occupancy into a global 2D map along estimated trajectories.

Input: ``trajectories.csv`` written by ``keyframe_backend`` (columns
raw_*, optimized_*, target_* and the BEV file of every frame).  Each frame's
occupied cells are transformed by its pose and accumulated in a world grid.
Panels: map from the ground-truth-integrated reference poses, from the raw
frontend, and from the backend.  Quality numbers compare each map with the
reference map (both built from the same BEV observations), so they measure
pose consistency only, not sensor quality.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .se2 import matrix, transform_points


def frame_points(path, cache={}):
    if path not in cache:
        meta = json.loads((Path(path).parent / 'metadata.json').read_text(encoding='utf-8'))['bev_config']
        with np.load(path, allow_pickle=False) as archive:
            bev = archive['bev'].astype(np.float32)
        rows, cols = np.nonzero(bev[0] > 0.5)
        res = float(meta['resolution_m'])
        cache[path] = np.column_stack((meta['forward_min_m'] + (rows + 0.5) * res,
                                       meta['left_min_m'] + (cols + 0.5) * res))
    return cache[path]


def accumulate(rows, prefix, stride):
    points = []
    rows = [r for r in rows if math.isfinite(float(r[f'{prefix}_x_m']))]
    for row in rows[::stride]:
        pose = matrix(float(row[f'{prefix}_x_m']), float(row[f'{prefix}_y_m']), float(row[f'{prefix}_yaw_rad']))
        points.append(transform_points(pose, frame_points(row['bev'])))
    return np.concatenate(points)


def occupancy(points, origin, shape, resolution, min_hits):
    index = np.floor((points - origin) / resolution).astype(int)
    keep = (index >= 0).all(1) & (index[:, 0] < shape[0]) & (index[:, 1] < shape[1])
    grid = np.zeros(shape)
    np.add.at(grid, (index[keep, 0], index[keep, 1]), 1)
    return grid, grid >= min_hits


def map_quality(grid_mask, reference_mask, origin, resolution):
    cells = np.argwhere(grid_mask) * resolution + origin
    ref = np.argwhere(reference_mask) * resolution + origin
    if not len(cells) or not len(ref):
        return {}
    d_to_ref, _ = cKDTree(ref).query(cells)
    d_from_ref, _ = cKDTree(cells).query(ref)
    return dict(occupied_cells=int(len(cells)), reference_cells=int(len(ref)),
                cell_ratio=len(cells) / len(ref),
                chamfer_m=float(0.5 * (d_to_ref.mean() + d_from_ref.mean())),
                within_10cm=float((d_to_ref <= 0.10).mean()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectories', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--resolution', type=float, default=0.05)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--min-hits', type=int, default=3)
    parser.add_argument('--title', default='')
    parser.add_argument('--labels', default='真值参考,前端,后端')
    parser.add_argument('--combine', action='store_true', help='all sessions in one map (shared world frame)')
    parser.add_argument('--subtitle', default='（起点对齐，无回到起点）')
    parser.add_argument('--lang', choices=('zh', 'en'), default='zh', help='language of figure text')
    parser.add_argument('--panels', default='target,raw,optimized', help='which panels to draw, e.g. raw,optimized')
    parser.add_argument('--no-quality', action='store_true', help='do not print map-quality numbers (e.g. no ground truth)')
    args = parser.parse_args(argv)
    if args.lang == 'en' and args.labels == '真值参考,前端,后端':
        args.labels = 'Reference (ground truth),Front end only,After loop closure'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for font in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC'):
        if font in {f.name for f in font_manager.fontManager.ttflist}:
            plt.rcParams['font.sans-serif'] = [font]
            break
    plt.rcParams['axes.unicode_minus'] = False
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    with open(args.trajectories, encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    labels = args.labels.split(',')
    report = {}
    if args.combine:
        for r in rows:
            r['segment'] = '0'
    for segment in sorted({r['segment'] for r in rows}, key=int):
        seg = [r for r in rows if r['segment'] == segment]
        clouds = {p: accumulate(seg, p, args.stride) for p in ('target', 'raw', 'optimized')}
        allpts = np.concatenate(list(clouds.values()))
        lo = np.percentile(allpts, 0.5, axis=0) - 0.5
        hi = np.percentile(allpts, 99.5, axis=0) + 0.5
        shape = tuple(np.ceil((hi - lo) / args.resolution).astype(int))
        grids = {p: occupancy(c, lo, shape, args.resolution, args.min_hits) for p, c in clouds.items()}
        quality = {p: map_quality(grids[p][1], grids['target'][1], lo, args.resolution) for p in ('raw', 'optimized')}
        report[f"{seg[0]['sequence']}/{segment}"] = quality
        names = ('target', 'raw', 'optimized')
        shown = [p for p in args.panels.split(',') if p in names]
        fig, axes = plt.subplots(1, len(shown), figsize=(6 * len(shown), 6.5), constrained_layout=True, squeeze=False)
        axes = axes[0]
        extent = (lo[1], hi[1], lo[0], hi[0])
        for ax, prefix in zip(axes, shown):
            label = labels[names.index(prefix)]
            grid = np.log1p(grids[prefix][0])
            ax.imshow(grid, origin='lower', extent=extent, cmap='Greys', vmax=np.percentile(grid[grid > 0], 98))
            # world x = forward is drawn vertically; y = left horizontally, flipped so left is left
            colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#17becf']
            for n, name in enumerate(dict.fromkeys(r.get('session', r['sequence']) for r in seg)):
                part = [r for r in seg if r.get('session', r['sequence']) == name
                        and math.isfinite(float(r[f'{prefix}_x_m']))]
                if part:
                    ax.plot([float(r[f'{prefix}_y_m']) for r in part], [float(r[f'{prefix}_x_m']) for r in part],
                            color=colors[n % len(colors)], lw=1.2, label=name if args.combine else None)
            if args.combine and prefix == shown[0]:
                ax.legend(fontsize=7, loc='best')
            if not args.combine:
                ax.plot(0, 0, 'o', color='#2ca02c')
            ax.invert_xaxis()
            ax.set_aspect('equal')
            q = {} if args.no_quality else quality.get(prefix, {})
            if args.lang == 'en':
                sub = '' if not q else f"\nChamfer distance to reference {q['chamfer_m']:.2f} m, {100 * q['within_10cm']:.0f}% within 10 cm"
                ax.set_xlabel('left / m')
                ax.set_ylabel('forward / m')
            else:
                sub = '' if not q else f"\n与参考地图倒角距离 {q['chamfer_m']:.2f} m，10 cm 内 {100 * q['within_10cm']:.0f}%"
                ax.set_xlabel('左向 / m')
                ax.set_ylabel('前向 / m')
            ax.set_title(label + sub)
        if args.lang == 'en':
            fig.suptitle(args.title, y=1.04)
        else:
            where = '全部段' if args.combine else f"{seg[0]['sequence']} 片段 {segment}"
            fig.suptitle(f"{args.title} {where}：BEV 占据拼接全局地图{args.subtitle}", y=1.04)
        fig.savefig(output / (f"map_all.png" if args.combine else f"map_{seg[0]['sequence']}_{segment}.png"),
                    dpi=140, bbox_inches='tight')
        plt.close(fig)
    (output / 'map_quality.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
