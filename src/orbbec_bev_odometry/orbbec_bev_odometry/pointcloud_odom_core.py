"""ROS-independent scan-to-submap point-cloud odometry pipeline."""

from dataclasses import dataclass
import time

import numpy as np

from .cloud_preprocessor import CloudPreprocessor, PreprocessConfig
from .local_submap import LocalSubmap
from .gicp_initializer import GicpInitializer
from .planar_icp import IcpConfig, PlanarPointToPlaneICP
from .se2 import components, inverse, matrix


@dataclass
class OdomResult:
    pose: np.ndarray
    accepted: bool
    rmse: float
    inlier_ratio: float
    correspondences: int
    processing_ms: float
    filtered_points: int


class PointCloudOdomCore:
    """Preprocess scans, register to a rolling submap, and integrate SE(2)."""

    def __init__(self, preprocess=None, icp=None, submap_scans=12,
                 stationary_translation=0.008, stationary_yaw=0.002,
                 yaw_gain=1.016):
        self.preprocessor = CloudPreprocessor(preprocess or PreprocessConfig())
        self.icp = PlanarPointToPlaneICP(icp or IcpConfig())
        self.submap = LocalSubmap(max_scans=submap_scans)
        self.pose = np.eye(3)
        self.map_pose = np.eye(3)
        self.last_delta = np.eye(3)
        self.last_scan = None
        self.last_cloud = None
        self.gicp = GicpInitializer()
        self.stationary_translation = stationary_translation
        self.stationary_yaw = stationary_yaw
        self.yaw_gain = yaw_gain

    def process(self, xyz):
        started = time.perf_counter()
        cloud = self.preprocessor.process(xyz)
        planar = cloud[:, :2]
        if not self.submap.scans:
            self.submap.add(planar, self.map_pose)
            self.last_scan = planar
            self.last_cloud = cloud
            return OdomResult(self.pose.copy(), False, np.inf, 0.0, 0,
                              (time.perf_counter() - started) * 1000.0, len(planar))
        predicted = self.map_pose @ self.last_delta
        target = self.submap.target_in_frame(predicted)
        frame_registration = self.gicp.register(cloud, self.last_cloud)
        if frame_registration is None:
            frame_registration = self.icp.register(
                planar, self.last_scan, initial=np.eye(3)
            )
        if not frame_registration.accepted:
            # A wide joint search is safe only between adjacent scans. Using it
            # against the whole submap can jump to repeated corridor geometry.
            recovered_frame = self.icp.register(planar, self.last_scan)
            if recovered_frame.accepted:
                frame_registration = recovered_frame
        self.last_scan = planar
        self.last_cloud = cloud
        initial = np.eye(3)
        frame_pose = None
        frame_moving = False
        if frame_registration.accepted:
            frame_pose = self.map_pose @ frame_registration.transform
            initial = inverse(predicted) @ frame_pose
            fdx, fdy, fdyaw = components(frame_registration.transform)
            frame_moving = (
                np.hypot(fdx, fdy) >= self.stationary_translation
                or abs(fdyaw) >= self.stationary_yaw
            )
        registration = self.icp.register(planar, target, initial=initial)
        if registration.accepted and frame_pose is not None:
            submap_pose = predicted @ registration.transform
            disagreement = inverse(frame_pose) @ submap_pose
            ex, ey, eyaw = components(disagreement)
            if np.hypot(ex, ey) > 0.10 or abs(eyaw) > np.deg2rad(5.0):
                registration = frame_registration
                new_pose = frame_pose
            else:
                new_pose = submap_pose
        if not registration.accepted and frame_pose is not None:
            # Preserve continuity through short submap degeneracies without a
            # wide search that can jump to a repeated corridor structure.
            registration = frame_registration
            new_pose = frame_pose
        elif not (registration.accepted and frame_pose is not None):
            new_pose = predicted @ registration.transform
        if registration.accepted:
            map_delta = inverse(self.map_pose) @ new_pose
            dx, dy, dyaw = components(map_delta)
            if frame_registration.accepted:
                odx, _, odyaw = components(frame_registration.transform)
                output_pose = self.pose @ matrix(odx, 0.0, odyaw * self.yaw_gain)
            else:
                output_pose = new_pose.copy()
            moved = True
            if (not frame_moving) or (
                np.hypot(dx, dy) < self.stationary_translation
                and abs(dyaw) < self.stationary_yaw
            ):
                new_pose = self.map_pose.copy()
                output_pose = self.pose.copy()
                dx = dy = dyaw = 0.0
                moved = False
            # Constant-velocity prediction is deliberately damped to avoid overshoot.
            self.last_delta = matrix(dx * 0.75, dy * 0.75, dyaw * 0.75)
            self.map_pose = new_pose
            self.pose = output_pose
            # Do not smear a changing scan into the same map pose while a
            # deadband is holding it; displacement must accumulate against the
            # unchanged submap and will then be released as one valid step.
            if moved:
                self.submap.add(planar, self.map_pose)
        else:
            self.last_delta = np.eye(3)
        return OdomResult(
            self.pose.copy(), registration.accepted, registration.rmse,
            registration.inlier_ratio, registration.correspondences,
            (time.perf_counter() - started) * 1000.0, len(planar)
        )
