"""Split from training/config.py (verbatim node moves). Part: data."""

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from app.vjepa_cowa_world_model.training.configs.common import NAVSIM_IMAGE_REQUIRE_AUTO, compute_tokens_per_frame

NAVSIM_DEFAULT_MAX_AGENTS = 256


def require_positive_navsim_max_agents(value: Any, *, field_name: str = "max_agents") -> int:
    """Return a strict positive NavSim agent capacity."""

    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")
    return int(value)


def resolve_navsim_root_max_agents(
    roots: Sequence[Mapping[str, Any]],
    *,
    default_max_agents: int = NAVSIM_DEFAULT_MAX_AGENTS,
    field_name: str = "NavSim roots",
) -> int:
    """Validate that all roots resolve to one stackable agent capacity."""

    default_capacity = require_positive_navsim_max_agents(
        default_max_agents,
        field_name="data.navsim.max_agents",
    )
    resolved = []
    for index, root in enumerate(roots):
        if not isinstance(root, Mapping):
            raise ValueError(f"{field_name}[{index}] must be a mapping, got {type(root).__name__}")
        resolved.append(
            require_positive_navsim_max_agents(
                root.get("max_agents", default_capacity),
                field_name=f"{field_name}[{index}].max_agents",
            )
        )
    if not resolved:
        return default_capacity
    capacities = set(resolved)
    if len(capacities) != 1:
        raise ValueError(
            f"{field_name} max_agents must resolve to the same capacity for collation, got {sorted(capacities)}"
        )
    return resolved[0]


def validate_navsim_cvoi_geometry_contract(
    roots: Sequence[Mapping[str, Any]],
    *,
    default_max_agents: int = NAVSIM_DEFAULT_MAX_AGENTS,
    default_load_agent_annotations: bool = True,
) -> int:
    """Validate the domain-specific CVoI NavSim geometry transport contract."""

    if not isinstance(default_load_agent_annotations, bool):
        raise ValueError(
            "default_load_agent_annotations must be a boolean, " f"got {default_load_agent_annotations!r}"
        )
    capacity = resolve_navsim_root_max_agents(
        roots,
        default_max_agents=default_max_agents,
        field_name="CVoI NavSim roots",
    )
    for index, root in enumerate(roots):
        if not isinstance(root, Mapping):
            raise ValueError(f"CVoI NavSim root {index} must be a mapping")
        domain = root.get("domain")
        load_agent_annotations = root.get("load_agent_annotations", default_load_agent_annotations)
        if not isinstance(load_agent_annotations, bool):
            raise ValueError(
                f"CVoI NavSim root {index} load_agent_annotations must be a boolean, "
                f"got {load_agent_annotations!r}"
            )
        if domain == "real":
            if load_agent_annotations is not True:
                raise ValueError(f"CVoI real NavSim root {index} requires load_agent_annotations=true")
        elif domain == "counterfactual":
            if load_agent_annotations is not False:
                raise ValueError(f"CVoI counterfactual NavSim root {index} requires load_agent_annotations=false")
        else:
            raise ValueError(f"CVoI NavSim root {index} domain must be 'real' or 'counterfactual', got {domain!r}")
    return capacity


validate_cvoi_navsim_geometry_contract = validate_navsim_cvoi_geometry_contract


@dataclass
class NavSimConfig:
    """NavSim 数据配置"""

    enabled: bool = False
    data_path: str = ""
    sensor_blobs_path: str = ""
    train_roots: List[Dict[str, Any]] = field(default_factory=list)
    balance_train_roots: bool = False
    val_roots: List[Dict[str, Any]] = field(default_factory=list)
    val_data_path: Optional[str] = None
    val_sensor_blobs_path: Optional[str] = None
    val_domain: Optional[str] = None
    val_annotation_selection: str = "all_valid"
    camera_name: str = "CAM_F0"
    camera_names: List[str] = field(default_factory=lambda: ["CAM_F0"])
    num_history_frames: Optional[int] = None
    image_require_policy: str = NAVSIM_IMAGE_REQUIRE_AUTO
    max_scenes: Optional[int] = None
    max_val_scenes: Optional[int] = None
    index_cache: bool = True  # 缓存场景索引到磁盘，避免每次启动重复扫描 pkl 文件
    window_stride: int = 1  # 训练集滑窗步长（帧），1=最大重叠，等于 frames_per_clip=无重叠
    val_window_stride: Optional[int] = None  # 验证集独立步长，None 时回退到 window_stride
    tail_seconds: Optional[float] = None  # 单 root 训练集只枚举最后 N 秒窗口；None=全视频
    val_tail_seconds: Optional[float] = None  # 验证集只枚举最后 N 秒窗口；None=全视频
    counterfactual_tail_seconds: Optional[float] = 5.0  # mixed train_roots 中 name=counterfactual 的默认后缀秒数
    max_frame_gap: int = 3  # 窗口内相邻 valid 帧之间允许的最大原始帧索引差，超出则丢弃该窗口
    max_agents: int = NAVSIM_DEFAULT_MAX_AGENTS
    load_agent_annotations: bool = True
    # 官方 scene-filter yaml 路径（navtrain/navtest）。设置即启用官方 token 锚定采样
    # （路径本身就是开关），与 window_stride≠1 / max_scenes 互斥；不设 = stride 滑窗。
    scene_filter_yaml: Optional[str] = None
    val_scene_filter_yaml: Optional[str] = None
    pose_overlay_path: Optional[str] = None
    val_pose_overlay_path: Optional[str] = None
    pose_overlay_coord_frame: str = "opencv_first_frame"
    pose_overlay_required: bool = False
    # 验证集反事实标注（val_annos.json）；train 侧按 root 配（train_roots[].annotations_path）
    val_annotations_path: Optional[str] = None
    val_annotations_drop_distorted: Optional[bool] = None


@dataclass
class Bench2DriveConfig:
    """Bench2Drive PKL 索引数据配置"""

    enabled: bool = False
    data_root: Optional[str] = None
    ann_file: Optional[str] = None
    val_ann_file: Optional[str] = None
    camera_name: str = "CAM_FRONT"
    base_fps: int = 10
    image_require_policy: str = NAVSIM_IMAGE_REQUIRE_AUTO
    max_scenes: Optional[int] = None
    max_val_scenes: Optional[int] = None
    window_stride: int = 1
    val_window_stride: Optional[int] = None
    max_frame_gap: int = 6
    max_agents: int = 50
    load_agent_annotations: bool = True
    command_dim: int = 6
    index_cache: bool = True
    index_cache_dir: Optional[str] = None
    verify_image_exists: bool = False
    max_load_retries: int = 5
    index_cache_wait_seconds: int = 300


@dataclass
class MongoRawConfig:
    """Mongo + raw clip 在线数据配置"""

    enabled: bool = False
    mongo_uri: Optional[str] = None
    mongo_uri_env: Optional[str] = None
    database: str = "e2e-data-platform-prod"
    collection: str = "clip"
    vehicle_type: Optional[str] = None
    vehicle_types: List[str] = field(default_factory=list)
    require_latest_available_revision: bool = True
    query_filter: Dict[str, Any] = field(default_factory=dict)
    start_index: int = 0
    end_index: Optional[int] = None
    max_clips: Optional[int] = None
    max_val_clips: Optional[int] = None
    val_ratio: float = 0.05
    split_seed: int = 0
    source_fps: int = 10
    base_fps: int = 5
    main_topic: str = "/main/ruby/lidar_points"
    pose_topic: str = "/pose/odom"
    match_topic: str = "/match"
    camera_topics: List[str] = field(default_factory=list)
    default_storage_root: str = "/path/to/mongo/default-storage"
    e2e_storage_root: str = "/path/to/mongo/e2e-storage"
    clipdata_storage_root: str = "/path/to/mongo/clipdata-storage"
    cache_size: int = 8
    max_retries: int = 5
    extra_camera_mappings: Dict[str, str] = field(default_factory=dict)
    record_cache_dir: Optional[str] = None  # 缓存目录，None 则不缓存
    record_cache_ttl: int = 604800  # 缓存过期时间（秒），默认 7 天
    blacklist_path: Optional[str] = None  # 坏 clip ID 黑名单 JSON 路径，None 则不使用


@dataclass
class DataConfig:
    """数据配置：数据集和加载器相关设置"""

    datasets: List[str] = field(default_factory=list)
    val_datasets: Optional[List[str]] = None
    dataset_fpcs: List[int] = field(default_factory=list)
    batch_size: int = 4
    tubelet_size: int = 2
    use_tubelet_repeat: bool = True
    fps: int = 5
    crop_size: Tuple[int, int] = (256, 256)
    patch_size: int = 16
    num_target_frames: int = 16
    pin_mem: bool = True  # pinned host memory speeds H2D transfer for video batches (byte-identical)
    # Default 8: video decode is the bottleneck and 1 starves the GPU (327/426 configs that set this use 8).
    # CAVEAT: per-worker RNG (seed_dataloader_worker) makes the __getitem__ training-window pick depend on
    # num_workers, so two runs with different num_workers are NOT byte-identical. For byte-exact verification
    # pin num_workers explicitly (the GPU smokes use 0). Proper fix (decouple the window RNG from num_workers)
    # tracked separately.
    num_workers: int = 8
    persistent_workers: bool = True
    camera_frame: bool = False
    camera_views: List[str] = field(default_factory=lambda: ["left_mp4_path"])
    stereo_view: bool = False
    navsim: Optional[NavSimConfig] = None
    bench2drive: Optional[Bench2DriveConfig] = None
    mongo_raw: Optional[MongoRawConfig] = None

    @property
    def dataset_path(self) -> Optional[str]:
        return self.datasets[0] if self.datasets else None

    @property
    def val_dataset_path(self) -> Optional[str]:
        return self.val_datasets[0] if self.val_datasets else None

    @property
    def max_num_frames(self) -> int:
        return max(self.dataset_fpcs) if self.dataset_fpcs else 16

    @property
    def crop_height(self) -> int:
        return self.crop_size[0]

    @property
    def crop_width(self) -> int:
        return self.crop_size[1]

    @property
    def tokens_per_frame(self) -> int:
        return compute_tokens_per_frame(self.crop_size, self.patch_size)


@dataclass
class DataAugConfig:
    """数据增强配置"""

    horizontal_flip: bool = False
    random_resize_aspect_ratio: List[float] = field(default_factory=lambda: [3 / 4, 4 / 3])
    random_resize_scale: List[float] = field(default_factory=lambda: [0.3, 1.0])
    motion_shift: bool = False
    reprob: float = 0.0
    auto_augment: bool = False
