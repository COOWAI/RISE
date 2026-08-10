"""NavSim pose overlay loading and conversion utilities."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PoseOverlayBatch:
    """Pose-overlay-derived NavSim training fields."""

    states: np.ndarray
    actions: np.ndarray
    ego_dynamics: np.ndarray


@dataclass(frozen=True)
class _PoseTable:
    frame_indices: np.ndarray
    translation: np.ndarray
    rotation: Optional[np.ndarray] = None


class PoseOverlayReader:
    """Load predicted poses and expose them as NavSim ego trajectory fields.

    Pose semantics are fixed to OpenCV coordinates with the first frame as the
    origin. The repository's ego trajectory convention is:

    ``ego_x = z_cv`` and ``ego_y = -x_cv``.
    """

    _VALID_COORD_FRAMES = {"opencv_first_frame"}
    _COUNTERFACTUAL_TXT_PATTERN = re.compile(r"^(?P<scene>.+)_cf_(?P<start>\d{6})_(?P<end>\d{6})$")
    _COUNTERFACTUAL_TXT_CAMERA = "CAM_F0"
    _COUNTERFACTUAL_TXT_POSE_FPS = 10.0
    _COUNTERFACTUAL_TIMELINE_FPS = 2.0
    _DEFAULT_COUNTERFACTUAL_TXT_START_SECONDS = 1.5

    def __init__(
        self,
        root: str | Path,
        *,
        coord_frame: str = "opencv_first_frame",
        required: bool = True,
        txt_start_seconds: float = _DEFAULT_COUNTERFACTUAL_TXT_START_SECONDS,
    ):
        self.root = Path(root)
        self.coord_frame = str(coord_frame)
        self.required = bool(required)
        self.txt_start_seconds = float(txt_start_seconds)
        if self.coord_frame not in self._VALID_COORD_FRAMES:
            raise ValueError(
                f"Unsupported pose overlay coord_frame={coord_frame!r}; "
                f"expected one of {sorted(self._VALID_COORD_FRAMES)}"
            )
        if self.required and not self.root.exists():
            raise FileNotFoundError(f"NavSim pose overlay root does not exist: {self.root}")
        if not np.isfinite(self.txt_start_seconds) or self.txt_start_seconds < 0.0:
            raise ValueError(
                "pose overlay txt_start_seconds must be a finite non-negative value, " f"got {txt_start_seconds!r}"
            )
        txt_start_row = self.txt_start_seconds * self._COUNTERFACTUAL_TXT_POSE_FPS
        if abs(txt_start_row - round(txt_start_row)) > 1e-6:
            raise ValueError(
                f"pose overlay txt_start_seconds={self.txt_start_seconds} is not aligned to "
                f"{self._COUNTERFACTUAL_TXT_POSE_FPS}Hz TXT poses"
            )
        self._cache: Dict[str, _PoseTable] = {}
        self._expected_scene_sha256: Optional[Dict[str, str]] = None

    def bind_expected_scene_sha256(self, expected_sha256_by_scene: Mapping[str, str]) -> None:
        """Bind immutable per-scene pose provenance before any overlay is loaded."""

        if not isinstance(expected_sha256_by_scene, Mapping) or not expected_sha256_by_scene:
            raise ValueError("expected pose overlay SHA256 mapping must be non-empty")
        normalized: Dict[str, str] = {}
        for raw_scene_name, raw_sha256 in expected_sha256_by_scene.items():
            scene_name = str(raw_scene_name)
            if not scene_name:
                raise ValueError("expected pose overlay SHA256 scene names must be non-empty")
            if type(raw_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", raw_sha256) is None:
                raise ValueError(f"expected pose overlay SHA256 for scene={scene_name!r} is invalid")
            normalized[scene_name] = raw_sha256
        if self._cache:
            raise RuntimeError("cannot bind expected pose overlay SHA256 after pose tables have been loaded")
        if self._expected_scene_sha256 is not None and self._expected_scene_sha256 != normalized:
            raise ValueError("pose overlay SHA256 mapping is already bound to different provenance")
        self._expected_scene_sha256 = normalized

    def get_pose_sequence(self, scene_name: str, frame_indices: Sequence[int]) -> _PoseTable:
        """Return a strict pose slice for ``scene_name`` and ``frame_indices``."""
        table = self._load_scene(scene_name)
        requested = np.asarray([int(idx) for idx in frame_indices], dtype=np.int64)
        if requested.ndim != 1:
            raise ValueError(f"frame_indices must be 1-D, got shape={requested.shape}")
        frame_to_pos = {int(frame_idx): pos for pos, frame_idx in enumerate(table.frame_indices.tolist())}
        missing = [int(frame_idx) for frame_idx in requested.tolist() if int(frame_idx) not in frame_to_pos]
        if missing:
            raise KeyError(
                f"NavSim pose overlay scene={scene_name!r} missing frame(s) {missing[:8]} " f"under root={self.root}"
            )
        take = np.asarray([frame_to_pos[int(frame_idx)] for frame_idx in requested.tolist()], dtype=np.int64)
        rotation = None if table.rotation is None else table.rotation[take]
        return _PoseTable(frame_indices=requested, translation=table.translation[take], rotation=rotation)

    def preload_scenes(self, scene_names: Sequence[str]) -> int:
        """Load pose tables for the provided scenes into the reader cache."""
        seen = set()
        for scene_name in scene_names:
            scene_key = str(scene_name)
            if scene_key in seen:
                continue
            self._load_scene(scene_key)
            seen.add(scene_key)
        return len(seen)

    def build_states_actions_ego_dynamics(
        self,
        scene_name: str,
        frame_indices: Sequence[int],
        *,
        action_dim: int,
        dt: float,
    ) -> PoseOverlayBatch:
        """Build ``states/actions/ego_dynamics`` from pose overlay.

        Parameters
        ----------
        scene_name    : NavSim scene/log name.
        frame_indices : reduced frame indices requested by the dataset.
        action_dim    : configured predictor/planner action dimension.
        dt            : seconds between adjacent reduced poses.
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        pose = self.get_pose_sequence(scene_name, frame_indices)
        xy_yaw = self._opencv_pose_to_ego_xy_yaw(pose.translation, pose.rotation)
        states = self._build_states_from_xy_yaw(xy_yaw, dt)
        actions = self._build_actions_from_states(states, int(action_dim))
        ego_dynamics = self._build_ego_dynamics_from_states(states, dt)
        return PoseOverlayBatch(states=states, actions=actions, ego_dynamics=ego_dynamics)

    def _load_scene(self, scene_name: str) -> _PoseTable:
        scene_key = str(scene_name)
        if scene_key in self._cache:
            return self._cache[scene_key]
        path = self._resolve_scene_file(scene_key)
        if path is None:
            raise FileNotFoundError(
                f"missing NavSim pose overlay scene={scene_key!r} under root={self.root}; "
                "expected <root>/<scene>/pred_pose.npz, <root>/<scene>.npz, JSON equivalents, "
                "or flat counterfactual *_CAM_F0_*_gen.txt pose files"
            )
        self._verify_scene_sha256(scene_key, path)
        if path.suffix == ".npz":
            table = self._load_npz(path)
        elif path.suffix == ".npy":
            table = self._load_npy(path)
        elif path.suffix == ".json":
            table = self._load_json(path)
        elif path.suffix == ".txt":
            table = self._load_txt(path)
        else:
            raise ValueError(f"Unsupported pose overlay file extension: {path}")
        self._validate_table(table, path)
        self._cache[scene_key] = table
        return table

    def _verify_scene_sha256(self, scene_name: str, path: Path) -> None:
        if self._expected_scene_sha256 is None:
            return
        expected = self._expected_scene_sha256.get(scene_name)
        if expected is None:
            raise ValueError(f"pose overlay scene={scene_name!r} has no bound sidecar SHA256 under root={self.root}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(
                f"pose overlay SHA256 mismatch for scene={scene_name!r}: "
                f"expected={expected}, actual={actual}, path={path}"
            )

    def _resolve_scene_file(self, scene_name: str) -> Optional[Path]:
        candidates = [
            self.root / scene_name / "pred_pose.npz",
            self.root / scene_name / "pred_pose.npy",
            self.root / scene_name / "pred_pose.json",
            self.root / scene_name / "pred_pose.txt",
            self.root / f"{scene_name}.npz",
            self.root / f"{scene_name}.npy",
            self.root / f"{scene_name}.json",
            self.root / f"{scene_name}.txt",
        ]
        cf_txt_name = self._counterfactual_txt_name(scene_name)
        if cf_txt_name is not None:
            candidates.append(self.root / cf_txt_name)
        for path in candidates:
            if path.is_file():
                return path
        return None

    @classmethod
    def _counterfactual_txt_name(cls, scene_name: str) -> Optional[str]:
        match = cls._COUNTERFACTUAL_TXT_PATTERN.match(str(scene_name))
        if match is None:
            return None
        return (
            f"{match.group('scene')}_{cls._COUNTERFACTUAL_TXT_CAMERA}_"
            f"{match.group('start')}_{match.group('end')}_gen.txt"
        )

    @staticmethod
    def _load_npz(path: Path) -> _PoseTable:
        data = np.load(path, allow_pickle=False)
        if "frame_indices" not in data or "translation" not in data:
            raise KeyError(f"pose overlay npz must contain frame_indices and translation: {path}")
        rotation = data["rotation"] if "rotation" in data else None
        return _PoseTable(
            frame_indices=np.asarray(data["frame_indices"], dtype=np.int64),
            translation=np.asarray(data["translation"], dtype=np.float64),
            rotation=None if rotation is None else np.asarray(rotation, dtype=np.float64),
        )

    @staticmethod
    def _load_npy(path: Path) -> _PoseTable:
        arr = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError(f"pose overlay npy must be [N, >=4] as frame,x,y,z..., got {arr.shape}: {path}")
        return _PoseTable(frame_indices=arr[:, 0].astype(np.int64), translation=arr[:, 1:4])

    @staticmethod
    def _load_json(path: Path) -> _PoseTable:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping) and "frames" in payload:
            payload = payload["frames"]
        if isinstance(payload, Mapping):
            items = sorted(payload.items(), key=lambda item: int(item[0]))
            frame_indices = np.asarray([int(k) for k, _ in items], dtype=np.int64)
            translation = np.asarray([v["translation"] if isinstance(v, Mapping) else v for _, v in items])
        elif isinstance(payload, list):
            frame_indices = np.asarray([int(item["frame_index"]) for item in payload], dtype=np.int64)
            translation = np.asarray([item["translation"] for item in payload], dtype=np.float64)
        else:
            raise ValueError(f"pose overlay json must be a mapping or list: {path}")
        return _PoseTable(frame_indices=frame_indices, translation=translation)

    def _load_txt(self, path: Path) -> _PoseTable:
        arr = np.asarray(np.loadtxt(path), dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"pose overlay txt must be 2-D, got {arr.shape}: {path}")
        if arr.shape[1] == 12:
            matrices = arr.reshape(arr.shape[0], 3, 4)
            translation = matrices[:, :, 3]
            rotation = matrices[:, :, :3]
            if path.name.endswith("_gen.txt"):
                row_start = int(round(self.txt_start_seconds * self._COUNTERFACTUAL_TXT_POSE_FPS))
                row_stride = int(round(self._COUNTERFACTUAL_TXT_POSE_FPS / self._COUNTERFACTUAL_TIMELINE_FPS))
                if row_start >= translation.shape[0]:
                    raise ValueError(
                        f"counterfactual pose txt is too short for txt_start_seconds="
                        f"{self.txt_start_seconds}s at "
                        f"{self._COUNTERFACTUAL_TXT_POSE_FPS}Hz: {path}"
                    )
                rows = np.arange(row_start, translation.shape[0], row_stride, dtype=np.int64)
                frame_indices = np.arange(rows.shape[0], dtype=np.int64)
                translation = translation[rows]
                rotation = rotation[rows]
            else:
                frame_indices = np.arange(translation.shape[0], dtype=np.int64)
            return _PoseTable(frame_indices=frame_indices, translation=translation, rotation=rotation)
        if arr.shape[1] >= 4:
            return _PoseTable(frame_indices=arr[:, 0].astype(np.int64), translation=arr[:, 1:4])
        raise ValueError(f"pose overlay txt must be [N, 12] or [N, >=4], got {arr.shape}: {path}")

    @staticmethod
    def _validate_table(table: _PoseTable, path: Path) -> None:
        if table.frame_indices.ndim != 1:
            raise ValueError(f"pose overlay frame_indices must be [N], got {table.frame_indices.shape}: {path}")
        if table.translation.ndim != 2 or table.translation.shape[1] != 3:
            raise ValueError(f"pose overlay translation must be [N, 3], got {table.translation.shape}: {path}")
        if table.translation.shape[0] != table.frame_indices.shape[0]:
            raise ValueError(
                "pose overlay frame_indices and translation length mismatch: "
                f"{table.frame_indices.shape[0]} vs {table.translation.shape[0]} at {path}"
            )
        if len(set(table.frame_indices.tolist())) != int(table.frame_indices.shape[0]):
            raise ValueError(f"pose overlay contains duplicate frame_indices: {path}")
        if not np.isfinite(table.translation).all():
            raise ValueError(f"pose overlay contains non-finite translation values: {path}")
        if table.rotation is not None:
            if table.rotation.shape[0] != table.frame_indices.shape[0]:
                raise ValueError(
                    f"pose overlay rotation length {table.rotation.shape[0]} "
                    f"!= frame count {table.frame_indices.shape[0]}: {path}"
                )
            if not np.isfinite(table.rotation).all():
                raise ValueError(f"pose overlay contains non-finite rotation values: {path}")

    @classmethod
    def _opencv_pose_to_ego_xy_yaw(cls, translation: np.ndarray, rotation: Optional[np.ndarray]) -> np.ndarray:
        ego_x = translation[:, 2]
        ego_y = -translation[:, 0]
        if rotation is not None:
            yaw = cls._yaw_from_rotation(rotation)
        else:
            yaw = cls._yaw_from_displacement(ego_x, ego_y)
        return np.stack([ego_x, ego_y, yaw], axis=1).astype(np.float32)

    @staticmethod
    def _yaw_from_rotation(rotation: np.ndarray) -> np.ndarray:
        rot = np.asarray(rotation, dtype=np.float64)
        if rot.ndim == 3 and rot.shape[1:] == (3, 3):
            matrices = rot
        elif rot.ndim == 2 and rot.shape[1] == 9:
            matrices = rot.reshape(rot.shape[0], 3, 3)
        elif rot.ndim == 2 and rot.shape[1] == 4:
            matrices = Rotation.from_quat(rot).as_matrix()
        elif rot.ndim == 2 and rot.shape[1] >= 3:
            return rot[:, 2].astype(np.float64)
        else:
            raise ValueError(f"rotation must be [N,3,3], [N,9], [N,4], or [N,>=3], got {rot.shape}")

        # OpenCV camera forward is +z and right is +x. Project camera forward
        # into repo ego xy: ego_x=z_cv, ego_y=-x_cv.
        forward_cv = matrices[:, :, 2]
        forward_ego_x = forward_cv[:, 2]
        forward_ego_y = -forward_cv[:, 0]
        return np.arctan2(forward_ego_y, forward_ego_x)

    @staticmethod
    def _yaw_from_displacement(ego_x: np.ndarray, ego_y: np.ndarray) -> np.ndarray:
        yaw = np.zeros_like(ego_x, dtype=np.float64)
        if ego_x.shape[0] <= 1:
            return yaw
        dx = np.diff(ego_x)
        dy = np.diff(ego_y)
        segment_yaw = np.arctan2(dy, dx)
        yaw[1:] = segment_yaw
        return yaw

    @staticmethod
    def _build_states_from_xy_yaw(xy_yaw: np.ndarray, dt: float) -> np.ndarray:
        states = np.zeros((xy_yaw.shape[0], 7), dtype=np.float32)
        states[:, 0] = xy_yaw[:, 0]
        states[:, 1] = xy_yaw[:, 1]
        states[:, 5] = xy_yaw[:, 2]
        dynamics = PoseOverlayReader._build_ego_dynamics_from_states(states, dt)
        states[:, 6] = np.linalg.norm(dynamics[:, :2], axis=1).astype(np.float32)
        return states

    @staticmethod
    def _build_actions_from_states(states: np.ndarray, action_dim: int) -> np.ndarray:
        if states.shape[0] < 2:
            raise ValueError("Need at least 2 overlay states to build actions")
        base = np.zeros((states.shape[0] - 1, max(action_dim, 3)), dtype=np.float32)
        for t in range(states.shape[0] - 1):
            dx_global = states[t + 1, 0] - states[t, 0]
            dy_global = states[t + 1, 1] - states[t, 1]
            yaw = states[t, 5]
            cos_h = np.cos(-yaw)
            sin_h = np.sin(-yaw)
            dx_ego = cos_h * dx_global - sin_h * dy_global
            dy_ego = sin_h * dx_global + cos_h * dy_global
            d_yaw = states[t + 1, 5] - states[t, 5]
            base[t, 0] = dx_ego
            base[t, 1] = dy_ego
            base[t, 2] = np.arctan2(np.sin(d_yaw), np.cos(d_yaw))
        return base[:, :action_dim]

    @staticmethod
    def _build_ego_dynamics_from_states(states: np.ndarray, dt: float) -> np.ndarray:
        dynamics = np.zeros((states.shape[0], 4), dtype=np.float32)
        if states.shape[0] <= 1:
            return dynamics
        delta_xy = states[1:, :2] - states[:-1, :2]
        velocity = delta_xy / float(dt)
        dynamics[1:, :2] = velocity
        dynamics[0, :2] = velocity[0]
        if states.shape[0] > 2:
            accel = (dynamics[1:, :2] - dynamics[:-1, :2]) / float(dt)
            dynamics[1:, 2:4] = accel
        return dynamics
