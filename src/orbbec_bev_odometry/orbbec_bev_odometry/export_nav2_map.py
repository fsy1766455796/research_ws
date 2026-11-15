"""Export a SLAM result as a ROS 2 Nav2 occupancy map (map.pgm + map.yaml).

Input: ``trajectories.csv`` from ``appearance_slam`` / ``loop_slam`` (world
poses per frame and the BEV file of the frame).  Every BEV cell is classified
with the height bins of the 9-channel BEV:

* obstacle evidence: any height bin 0.25-1.8 m (channels 6, 7, 8);
* free evidence: ray casting.  The forward camera sees little floor, so for
  every 1 deg bearing inside the camera field of view the cells between the
  camera and the nearest obstacle cell are marked free.

Evidence is accumulated per map cell in the world frame; a cell is occupied
with enough obstacle hits that also make up a minimum share of all hits,
free with enough free hits, otherwise unknown.  Map values follow
``map_server`` trinary mode: 0 occupied, 254 free, 205 unknown.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def frame_cells(path, max_range):
    meta = json.loads((Path(path).parent / 'metadata.json').read_text(encoding='utf-8'))['bev_config']
    with np.load(path, allow_pickle=False) as archive:
        bev = archive['bev'].astype(np.float32)
    res = float(meta['resolution_m'])
    obstacle = (bev[6] > 0.5) | (bev[7] > 0.5) | (bev[8] > 0.5)
    rows, cols = np.nonzero(obstacle)
    pts = np.column_stack((meta['forward_min_m'] + (rows + 0.5) * res, meta['left_min_m'] + (cols + 0.5) * res))
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) <= max_range]
    # ray casting from the camera (base_link ~ (0.23, 0)) to the nearest obstacle per 1 deg bearing
    cam = np.array([0.23, 0.0])
    rel = pts - cam
    ok = rel[:, 0] > 0.05
    bearing = np.degrees(np.arctan2(rel[ok, 1], rel[ok, 0]))
    dist = np.hypot(rel[ok, 0], rel[ok, 1])
    free = []
    half_fov = 34.0
    for b in range(-int(half_fov), int(half_fov) + 1):
        sel = np.abs(bearing - b) <= 0.5
        if not sel.any():
            continue
        d = dist[sel].min() - 1.5 * res
        if d <= 0:
            continue
        r = np.arange(0.2, d, res / 2)
        a = math.radians(b)
        free.append(np.column_stack((cam[0] + r * math.cos(a), cam[1] + r * math.sin(a))))
    free = np.concatenate(free) if free else np.zeros((0, 2))
    return pts, free


def to_world(pose, pts):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.column_stack((c * pts[:, 0] - s * pts[:, 1] + pose[0], s * pts[:, 0] + c * pts[:, 1] + pose[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--trajectories', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--pose', default='optimized', help='optimized | target | raw')
    parser.add_argument('--resolution', type=float, default=0.05)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--max-range', type=float, default=4.0)
    parser.add_argument('--min-occupied-hits', type=int, default=3)
    parser.add_argument('--min-occupied-share', type=float, default=0.3)
    parser.add_argument('--min-free-hits', type=int, default=2)
    parser.add_argument('--title', default='')
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    with open(args.trajectories, encoding='utf-8-sig', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if math.isfinite(float(r[f'{args.pose}_x_m']))]
    occ_pts, free_pts, traj = [], [], {}
    for r in rows[::args.stride]:
        pose = [float(r[f'{args.pose}_{k}']) for k in ('x_m', 'y_m', 'yaw_rad')]
        o, f = frame_cells(r['bev'], args.max_range)
        occ_pts.append(to_world(pose, o))
        free_pts.append(to_world(pose, f))
    for r in rows:
        traj.setdefault(r.get('session', r['sequence']), []).append(
            (float(r[f'{args.pose}_x_m']), float(r[f'{args.pose}_y_m'])))
    occ_pts, free_pts = np.concatenate(occ_pts), np.concatenate(free_pts)
    allp = np.concatenate((occ_pts, free_pts))
    lo = np.floor((np.percentile(allp, 0.2, axis=0) - 0.5) / args.resolution) * args.resolution
    hi = np.percentile(allp, 99.8, axis=0) + 0.5
    width, height = (np.ceil((hi - lo) / args.resolution).astype(int))
    def hist(p):
        ix = np.floor((p - lo) / args.resolution).astype(int)
        keep = (ix[:, 0] >= 0) & (ix[:, 0] < width) & (ix[:, 1] >= 0) & (ix[:, 1] < height)
        h = np.zeros((width, height))
        np.add.at(h, (ix[keep, 0], ix[keep, 1]), 1)
        return h
    occ, free = hist(occ_pts), hist(free_pts)
    occupied = (occ >= args.min_occupied_hits) & (occ >= args.min_occupied_share * (occ + free))
    is_free = (free >= args.min_free_hits) & ~occupied
    grid = np.full((width, height), 205, np.uint8)
    grid[is_free] = 254
    grid[occupied] = 0
    image = np.flipud(grid.T)                  # row 0 = max y (map_server convention)
    from PIL import Image
    Image.fromarray(image, mode='L').save(output / 'map.pgm')
    (output / 'map.yaml').write_text(
        'image: map.pgm\nmode: trinary\n'
        f'resolution: {args.resolution}\norigin: [{lo[0]:.3f}, {lo[1]:.3f}, 0.0]\n'
        'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n', encoding='utf-8')
    stats = dict(width=int(width), height=int(height), resolution=args.resolution,
                 origin=[float(lo[0]), float(lo[1])], occupied_cells=int(occupied.sum()),
                 free_cells=int(is_free.sum()), unknown_cells=int(grid.size - occupied.sum() - is_free.sum()),
                 pose=args.pose, source=str(Path(args.trajectories).resolve()))
    (output / 'map_info.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
    # preview
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for font in ('Microsoft YaHei', 'SimHei'):
        if font in {f.name for f in font_manager.fontManager.ttflist}:
            plt.rcParams['font.sans-serif'] = [font]
            break
    fig, ax = plt.subplots(figsize=(8, 8 * height / width + 0.6))
    ax.imshow(image, cmap='gray', vmin=0, vmax=255,
              extent=(lo[0], lo[0] + width * args.resolution, lo[1], lo[1] + height * args.resolution))
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#17becf']
    for n, (name, pts) in enumerate(traj.items()):
        p = np.array(pts)
        ax.plot(p[:, 0], p[:, 1], color=colors[n % len(colors)], lw=1, label=name)
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlabel('x / m')
    ax.set_ylabel('y / m')
    ax.set_title(f"{args.title}\nNav2 栅格地图（黑=障碍，白=可通行，灰=未知），{args.resolution} m/格")
    fig.savefig(output / 'preview.png', dpi=130, bbox_inches='tight')
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
