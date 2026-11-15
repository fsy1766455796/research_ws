"""Convert a ROS 2 bag (colour + colour-aligned depth) into the OpenLORIS-style layout the offline pipeline reads.

Output
  <out-root>/<scene>/<seq>/color/<stamp>.jpg, color.txt, groundtruth.txt (identity poses: there is NO
  ground truth, the file only lets the existing tools run), sensors.yaml, trans_matrix.yaml
  <depth-root>/<seq>/color/<stamp>.npy   sensor depth in metres (float16, half resolution, 0 = invalid),
  stored where the pipeline expects monocular depth so the same code path runs unchanged.

The camera-to-floor geometry is not in the bag. Roll, pitch and height are estimated from the floor
plane seen in the depth images; heading is assumed to be straight ahead.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader


def floor_from_depth(samples, k):
    """Median floor normal (camera optical frame, pointing up) and camera height from depth samples."""
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    normals, heights = [], []
    for z in samples:
        h, w = z.shape
        v, u = np.mgrid[0:h:2, 0:w:2]
        zz = z[::2, ::2].astype(np.float64)
        ok = (zz > 0.4) & (zz < 5.0) & (v > h * 0.55)
        if ok.sum() < 3000:
            continue
        p = np.column_stack(((u[ok] - cx) / fx * zz[ok], (v[ok] - cy) / fy * zz[ok], zz[ok]))
        best = None
        rng = np.random.default_rng(0)
        for _ in range(120):
            a, b, c = p[rng.choice(len(p), 3, replace=False)]
            n = np.cross(b - a, c - a)
            if np.linalg.norm(n) < 1e-6:
                continue
            n /= np.linalg.norm(n)
            if abs(n[1]) < 0.8:                 # the floor normal is roughly the image "up" direction
                continue
            inl = np.abs((p - a) @ n) < 0.02
            if best is None or inl.sum() > best.sum():
                best = inl
        if best is None or best.sum() < 0.3 * len(p):
            continue
        q = p[best]
        c0 = q.mean(0)
        n = np.linalg.svd(q - c0, full_matrices=False)[2][2]
        if n[1] > 0:
            n = -n                              # optical y points down, so "up" has negative y
        normals.append(n)
        heights.append(abs(c0 @ n))
    if len(normals) < 5:
        return None
    n = np.median(np.array(normals), 0)
    return n / np.linalg.norm(n), float(np.median(heights)), len(normals)


def base_from_floor(n_up, height):
    """T_base_color: base x = camera forward projected on the floor, z = floor normal, origin on the floor."""
    z = n_up
    fwd = np.array([0.0, 0.0, 1.0])
    x = fwd - (fwd @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    t = np.eye(4)
    t[:3, :3] = np.vstack((x, y, z))
    t[2, 3] = height
    return t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bag', required=True)
    ap.add_argument('--sequence', required=True)
    ap.add_argument('--scene', default='lab')
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--depth-root', required=True)
    ap.add_argument('--color-topic', default='/camera/color/image_raw')
    ap.add_argument('--depth-topic', default='/camera/depth/image_raw')
    ap.add_argument('--info-topic', default='/camera/color/camera_info')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--fallback-height', type=float, default=0.5)
    args = ap.parse_args(argv)
    seq_dir = Path(args.out_root) / args.scene / args.sequence
    (seq_dir / 'color').mkdir(parents=True, exist_ok=True)
    ddir = Path(args.depth_root) / args.sequence / 'color'
    ddir.mkdir(parents=True, exist_ok=True)
    k = dist = size = None
    maps = None
    pending_depth = {}
    colors, samples = [], []
    n_depth = 0

    def stamp(m):
        return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9

    with AnyReader([Path(args.bag)]) as r:
        conns = [c for c in r.connections if c.topic in (args.color_topic, args.depth_topic, args.info_topic)]
        for c, _, raw in r.messages(connections=conns):
            m = r.deserialize(raw, c.msgtype)
            if c.topic == args.info_topic:
                if k is None:
                    k = np.array(m.k, dtype=np.float64).reshape(3, 3)
                    dist = np.array(list(m.d), dtype=np.float64)
                    size = (int(m.width), int(m.height))
                    if dist.size and np.abs(dist).max() > 1e-6:
                        maps = cv2.initUndistortRectifyMap(k, dist, None, k, size, cv2.CV_32FC1)
                continue
            if k is None:
                continue
            ts = stamp(m)
            if c.topic == args.depth_topic:
                d = np.frombuffer(m.data, dtype=np.uint16).reshape(m.height, m.width)
                if maps is not None and (m.width, m.height) == size:
                    d = cv2.remap(d, maps[0], maps[1], cv2.INTER_NEAREST)
                pending_depth[ts] = d
                for old in [t for t in pending_depth if t < ts - 0.5]:
                    pending_depth.pop(old)
                continue
            enc = m.encoding.lower()
            img = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.width, -1)
            if enc.startswith('rgb'):
                img = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)
            elif enc.startswith('bgra'):
                img = img[..., :3]
            if maps is not None:
                img = cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR)
            near = min(pending_depth, key=lambda t: abs(t - ts), default=None)
            if near is None or abs(near - ts) > 0.02:
                continue
            d = pending_depth[near]
            name = f'{ts:.6f}'
            cv2.imwrite(str(seq_dir / 'color' / f'{name}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
            dm = d.astype(np.float32) / 1000.0
            if (dm.shape[1], dm.shape[0]) != size:
                dm = cv2.resize(dm, size, interpolation=cv2.INTER_NEAREST)
            half = dm[::2, ::2]
            np.save(ddir / f'{name}.npy', half.astype(np.float16))
            colors.append((ts, f'color/{name}.jpg'))
            n_depth += 1
            if len(colors) % 30 == 1:
                samples.append(dm)
            if args.limit and len(colors) >= args.limit:
                break
    if not colors:
        raise SystemExit('no synchronised colour/depth frames found')
    colors.sort()
    with open(seq_dir / 'color.txt', 'w', encoding='utf-8') as f:
        f.write('# timestamp filename\n')
        f.writelines(f'{t:.6f} {p}\n' for t, p in colors)
    with open(seq_dir / 'groundtruth.txt', 'w', encoding='utf-8') as f:
        f.write('# NO GROUND TRUTH: identity poses so that the offline tools run\n')
        f.writelines(f'{t:.6f} 0 0 0 0 0 0 1\n' for t, _ in colors)
    floor = floor_from_depth(samples, k)
    if floor is None:
        n_up, height, used = np.array([0.0, -1.0, 0.0]), args.fallback_height, 0
    else:
        n_up, height, used = floor
    t_bc = base_from_floor(n_up, height)
    fs = cv2.FileStorage(str(seq_dir / 'sensors.yaml'), cv2.FILE_STORAGE_WRITE)
    fs.startWriteStruct('d400_color_optical_frame', cv2.FileNode_MAP)
    fs.write('intrinsics', np.array([[k[0, 0], k[0, 2], k[1, 1], k[1, 2]]]))
    fs.write('width', size[0])
    fs.write('height', size[1])
    fs.endWriteStruct()
    fs.release()
    fs = cv2.FileStorage(str(seq_dir / 'trans_matrix.yaml'), cv2.FILE_STORAGE_WRITE)
    fs.startWriteStruct('trans_matrix', cv2.FileNode_SEQ)
    fs.startWriteStruct('', cv2.FileNode_MAP)
    fs.write('parent_frame', 'base_link')
    fs.write('child_frame', 'd400_color_optical_frame')
    fs.write('matrix', t_bc)
    fs.endWriteStruct()
    fs.endWriteStruct()
    fs.release()
    pitch = math.degrees(math.asin(-(t_bc[:3, :3] @ np.array([0, 0, 1.0]))[2]))
    info = dict(bag=str(args.bag), sequence=args.sequence, frames=len(colors), duration_s=colors[-1][0] - colors[0][0],
                image_size=size, intrinsics_fx_fy_cx_cy=[k[0, 0], k[1, 1], k[0, 2], k[1, 2]],
                distortion=dist.tolist(), undistorted=maps is not None, depth='sensor depth aligned to colour, half resolution',
                floor_frames_used=used, camera_height_m=height, camera_pitch_down_deg=pitch,
                ground_truth='none (identity placeholder)')
    (seq_dir / 'conversion.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info))


if __name__ == '__main__':
    main()
