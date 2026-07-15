"""Global BEV map fusion, correction, and ROS message conversion."""

import math

import cv2
import numpy as np
from nav_msgs.msg import OccupancyGrid


class GlobalBEVMap:
    """Maintain a fixed BEV map and optionally correct translation drift."""

    def __init__(self, map_size_m, resolution_m, x_min_m, z_min_m,
                 search_radius_m, min_score, warmup_frames):
        self.resolution_m = float(resolution_m)
        size_px = int(round(float(map_size_m) / self.resolution_m))
        self.global_map = np.zeros((size_px, size_px), dtype=np.uint8)
        self.origin_x = size_px // 2
        self.origin_y = size_px // 2
        self.x_min_m = float(x_min_m)
        self.z_min_m = float(z_min_m)
        self.search_radius_px = max(
            1, int(round(float(search_radius_m) / self.resolution_m)))
        self.min_score = float(min_score)
        self.warmup_frames = int(warmup_frames)
        self.frame_count = 0

    def _top_left(self, shape, total_x, total_z, lateral_sign):
        height, width = shape
        pose_x = self.origin_x + int(round(float(total_x) / self.resolution_m))
        pose_y = self.origin_y - int(round(float(total_z) / self.resolution_m))
        robot_u = int(round(-self.x_min_m / self.resolution_m))
        if float(lateral_sign) < 0.0:
            robot_u = width - 1 - robot_u
        robot_v = int(round((height - 1) + self.z_min_m / self.resolution_m))
        return pose_x - robot_u, pose_y - robot_v, robot_u, robot_v

    def _search_image(self, shape, predicted_left, predicted_top):
        height, width = shape
        margin = self.search_radius_px
        search = np.zeros((height + 2 * margin, width + 2 * margin), dtype=np.uint8)
        map_left = predicted_left - margin
        map_top = predicted_top - margin
        src_x0 = max(0, map_left)
        src_y0 = max(0, map_top)
        src_x1 = min(self.global_map.shape[1], map_left + search.shape[1])
        src_y1 = min(self.global_map.shape[0], map_top + search.shape[0])
        if src_x0 < src_x1 and src_y0 < src_y1:
            dst_x0 = src_x0 - map_left
            dst_y0 = src_y0 - map_top
            search[dst_y0:dst_y0 + src_y1 - src_y0,
                   dst_x0:dst_x0 + src_x1 - src_x0] = (
                self.global_map[src_y0:src_y1, src_x0:src_x1])
        return search, map_left, map_top

    def _fuse_pose(self, bev, total_x, total_z, yaw_rad, lateral_sign):
        height, _ = bev.shape
        pose_x = self.origin_x + float(total_x) / self.resolution_m
        pose_y = self.origin_y - float(total_z) / self.resolution_m
        robot_u = -self.x_min_m / self.resolution_m
        robot_v = (height - 1) + self.z_min_m / self.resolution_m
        cos_yaw = math.cos(float(yaw_rad))
        sin_yaw = math.sin(float(yaw_rad))
        lateral_sign = -1.0 if float(lateral_sign) < 0.0 else 1.0
        transform = np.array([
            [lateral_sign * cos_yaw, -sin_yaw,
             pose_x - lateral_sign * cos_yaw * robot_u + sin_yaw * robot_v],
            [lateral_sign * sin_yaw, cos_yaw,
             pose_y - lateral_sign * sin_yaw * robot_u - cos_yaw * robot_v],
        ], dtype=np.float32)
        warped = cv2.warpAffine(
            bev, transform,
            (self.global_map.shape[1], self.global_map.shape[0]),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0)
        np.maximum(self.global_map, warped, out=self.global_map)

    def update(self, bev, total_x, total_z, yaw_rad=0.0,
               enable_correction=False, lateral_sign=1.0):
        """Fuse a frame; map correction is opt-in and translation-only."""
        if bev.dtype != np.uint8 or bev.ndim != 2:
            raise ValueError("bev must be a uint8 single-channel image")
        matching_bev = np.fliplr(bev) if lateral_sign < 0.0 else bev
        left, top, robot_u, robot_v = self._top_left(
            matching_bev.shape, total_x, total_z, lateral_sign)
        corrected_x = float(total_x)
        corrected_z = float(total_z)
        score = 0.0
        valid_mask = (matching_bev > 0).astype(np.uint8) * 255
        if (enable_correction and self.frame_count >= self.warmup_frames
                and np.count_nonzero(valid_mask) > 20):
            search, map_left, map_top = self._search_image(
                matching_bev.shape, left, top)
            scores = cv2.matchTemplate(
                search, matching_bev, cv2.TM_CCORR_NORMED, mask=valid_mask)
            scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
            _, score, _, best_loc = cv2.minMaxLoc(scores)
            score = float(score)
            if score >= self.min_score:
                pose_x = map_left + int(best_loc[0]) + robot_u
                pose_y = map_top + int(best_loc[1]) + robot_v
                corrected_x = (pose_x - self.origin_x) * self.resolution_m
                corrected_z = (self.origin_y - pose_y) * self.resolution_m
        self._fuse_pose(bev, corrected_x, corrected_z, yaw_rad, lateral_sign)
        self.frame_count += 1
        return corrected_x, corrected_z, score

    def to_occupancy_grid(self, stamp, frame_id):
        """Convert the accumulated map to an RViz OccupancyGrid."""
        grid = np.fliplr(self.global_map.T)
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.info.map_load_time = stamp
        msg.info.resolution = self.resolution_m
        msg.info.width = int(grid.shape[1])
        msg.info.height = int(grid.shape[0])
        msg.info.origin.position.x = (
            self.origin_y - self.global_map.shape[0] + 1) * self.resolution_m
        msg.info.origin.position.y = -self.origin_x * self.resolution_m
        msg.info.origin.orientation.w = 1.0
        scaled = np.rint(grid.astype(np.float32) * (100.0 / 255.0))
        msg.data = np.clip(scaled, 0, 100).astype(np.int8).ravel().tolist()
        return msg
