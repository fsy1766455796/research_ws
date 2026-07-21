"""Checkpoint loading and NumPy inference adapter for the ROS front end."""

from pathlib import Path

import numpy as np

from .multichannel_bev import MultiChannelBEVConfig, points_to_multichannel_bev


class LearnedBEVRuntime:
    """Own a trained model and expose one-pair metric SE(2) inference."""

    def __init__(self, checkpoint_path, device="auto"):
        """Load a training checkpoint onto the selected torch device."""
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "PyTorch is required when odometry_backend=learned"
            ) from error
        from .learned_bev_model import LearnedBEVOdometry

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                device = "mps"
            else:
                device = "cpu"
        self.torch = torch
        self.device = torch.device(device)
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Learned BEV checkpoint not found: {path}")
        try:
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.model = LearnedBEVOdometry(**checkpoint["model_config"])
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device).eval()

        config_values = dict(checkpoint.get("bev_config") or {})
        if "height_bins_m" in config_values:
            config_values["height_bins_m"] = tuple(
                config_values["height_bins_m"]
            )
        self.bev_config = MultiChannelBEVConfig(**config_values)
        expected_channels = int(checkpoint["model_config"]["input_channels"])
        if len(self.bev_config.channel_names) != expected_channels:
            raise ValueError(
                "Checkpoint BEV configuration does not match model channels: "
                f"{len(self.bev_config.channel_names)} != {expected_channels}"
            )

    def make_bev(self, robot_points_m):
        """Rasterize base-frame points with the checkpoint BEV config."""
        return points_to_multichannel_bev(
            np.asarray(robot_points_m, dtype=np.float32), self.bev_config
        )

    def estimate(self, previous_bev, current_bev):
        """Return ``[forward, left, yaw]`` and correlation confidence."""
        torch = self.torch
        previous = torch.from_numpy(
            np.asarray(previous_bev, dtype=np.float32)
        ).unsqueeze(0).to(self.device)
        current = torch.from_numpy(
            np.asarray(current_bev, dtype=np.float32)
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(previous, current)
        pose = output["pose"][0].detach().cpu().numpy()
        confidence = float(output["confidence"][0].detach().cpu())
        return pose.astype(np.float64), confidence
