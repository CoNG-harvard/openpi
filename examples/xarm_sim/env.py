import os
from dataclasses import dataclass, field

import numpy as np
from openpi_client.runtime import environment as _environment
from scipy.spatial.transform import Rotation as R
from typing_extensions import override


@dataclass
class EnvConfig:
    headless: bool = True
    physics_dt: float = 1.0 / 60.0
    rendering_dt: float = 1.0 / 60.0
    stage_units_in_meters: float = 1.0
    xarm_usd_path: str = os.environ.get("XARM_USD_PATH", "")
    table_usd_path: str = os.environ.get("TABLE_USD_PATH", "")
    table_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.5]))
    table_scale: np.ndarray = field(default_factory=lambda: np.array([1.5, 0.8, 0.05]))
    table_color: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5]))
    camera_eye: list[float] = field(default_factory=lambda: [2.0, 0.9, 1.2])
    camera_target: list[float] = field(default_factory=lambda: [1.1, 0.4, 0.9])
    light_intensity: float = 1200.0
    wrist_camera_warmup_steps: int = 5
    wrist_camera_resolution: tuple[int, int] = (320, 180)
    wrist_camera_translation: np.ndarray = field(
        default_factory=lambda: np.array([-0.0753, 0.0287, 0.0233])
    )
    wrist_camera_rpy: np.ndarray = field(
        default_factory=lambda: np.array([-0.3256, 0.0062, -1.5629])
    )
    wrist_camera_name: str = "wrist_camera"
    wrist_camera_prim_name: str = "WristCamera"
    wrist_link_name: str = "link7"
    wrist_image_left_key: str = "wrist_image_left"
    left_camera_resolution: tuple[int, int] = (320, 180)
    right_camera_resolution: tuple[int, int] = (320, 180)
    left_camera_translation: np.ndarray = field(
        default_factory=lambda: np.array([0.4155, 0.4775, 0.4114])
    )
    left_camera_rpy: np.ndarray = field(
        default_factory=lambda: np.array([-2.2160, -0.0226, -2.9637])
    )
    right_camera_translation: np.ndarray = field(
        default_factory=lambda: np.array([0.2596, -0.5001, 0.4369])
    )
    right_camera_rpy: np.ndarray = field(
        default_factory=lambda: np.array([-2.2773, 0.3362, -0.5369])
    )
    left_camera_name: str = "left_camera"
    right_camera_name: str = "right_camera"
    left_camera_prim_name: str = "LeftCamera"
    right_camera_prim_name: str = "RightCamera"
    exterior_image_1_left_key: str = "exterior_image_1_left"
    exterior_image_2_left_key: str = "exterior_image_2_left"
    ground_plane_prim_path: str = "/World/GroundPlane"
    ground_plane_name: str = "ground_plane"
    table_prim_path: str = "/World/Table"
    table_name: str = "table"
    cube_prim_path: str = "/World/TargetCube"
    cube_name: str = "target_cube"
    cube_position: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.0, 0.55]))
    cube_scale: np.ndarray = field(default_factory=lambda: np.array([0.04, 0.04, 0.04]))
    cube_color: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    light_prim_path: str = "/World/KeyLight"
    xarm_prim_path: str = "/World/XArm"
    xarm_usd_default: str = "Isaac/Robots/Ufactory/xarm7/xarm7.usd"
    xarm_placeholder_size: float = 0.1
    xarm_placeholder_name: str = "xarm_placeholder"
    xarm_placeholder_child_name: str = "Placeholder"


class XArmIsaacEnvironment(_environment.Environment):
    """A minimal Isaac Sim environment for an xArm pick task."""

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self._step_count = 0
        self._last_action = np.zeros(7, dtype=np.float64)
        self._last_action_dict = self._default_action_dict()
        try:
            from isaacsim import SimulationApp
        except Exception as exc:
            try:
                from omni.isaac.kit import SimulationApp
            except Exception:
                raise ImportError(
                    "Failed to import SimulationApp. Ensure Isaac Sim is installed and the "
                    "environment is launched via the Isaac Sim Python entrypoint."
                ) from exc

        self._simulation_app = SimulationApp({"headless": cfg.headless})

        from omni.isaac.core import World
        from omni.isaac.core.objects import FixedCuboid, GroundPlane
        from omni.isaac.core.prims import XFormPrim
        from omni.isaac.core.utils.prims import is_prim_path_valid
        from omni.isaac.core.utils.stage import add_reference_to_stage, set_stage_units
        from omni.isaac.core.utils.viewports import set_camera_view
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.sensor import Camera
        from pxr import Gf, Usd, UsdGeom, UsdLux

        set_stage_units(cfg.stage_units_in_meters)
        self._world = World(physics_dt=cfg.physics_dt, rendering_dt=cfg.rendering_dt)
        self._ground = GroundPlane(prim_path=cfg.ground_plane_prim_path, name=cfg.ground_plane_name)

        assets_root = get_assets_root_path()
        if not assets_root:
            raise RuntimeError(
                "Isaac assets root is not available. Provide --xarm-usd with an absolute or "
                "omniverse:// path, or configure the Isaac assets root."
            )

        use_default_table = not cfg.table_usd_path
        if cfg.table_usd_path:
            table_usd = self._resolve_asset_path(cfg.table_usd_path, assets_root)
            add_reference_to_stage(table_usd, cfg.table_prim_path)
        else:
            self._world.scene.add(
                FixedCuboid(
                    prim_path=cfg.table_prim_path,
                    name=cfg.table_name,
                    position=cfg.table_position,
                    scale=cfg.table_scale,
                    color=cfg.table_color,
                )
            )

        self._world.scene.add(
            FixedCuboid(
                prim_path=cfg.cube_prim_path,
                name=cfg.cube_name,
                position=cfg.cube_position,
                scale=cfg.cube_scale,
                color=cfg.cube_color,
            )
        )

        light = UsdLux.DistantLight.Define(self._world.stage, cfg.light_prim_path)
        light.CreateIntensityAttr(cfg.light_intensity)

        self._xarm_prim_path = cfg.xarm_prim_path
        xarm_usd = cfg.xarm_usd_path or self._join_assets_path(assets_root, cfg.xarm_usd_default)
        xarm_usd = self._resolve_asset_path(xarm_usd, assets_root)
        if xarm_usd:
            add_reference_to_stage(xarm_usd, self._xarm_prim_path)
        else:
            XFormPrim(prim_path=self._xarm_prim_path, name=cfg.xarm_placeholder_name)
            placeholder = UsdGeom.Cube.Define(
                self._world.stage,
                f"{self._xarm_prim_path}/{cfg.xarm_placeholder_child_name}",
            )
            placeholder.CreateSizeAttr(cfg.xarm_placeholder_size)

        if not is_prim_path_valid(self._xarm_prim_path):
            raise RuntimeError(f"XArm prim path missing after stage setup (usd={xarm_usd}).")

        self._wrist_camera = None
        self._left_camera = None
        self._right_camera = None
        wrist_parent_path = self._find_wrist_parent_path(Usd, self._xarm_prim_path, cfg.wrist_link_name)
        wrist_camera_path = f"{wrist_parent_path}/{cfg.wrist_camera_prim_name}"
        self._wrist_camera = Camera(
            prim_path=wrist_camera_path,
            name=cfg.wrist_camera_name,
            resolution=cfg.wrist_camera_resolution,
            frequency=1.0 / cfg.rendering_dt,
        )
        self._wrist_camera.set_local_pose(
            translation=cfg.wrist_camera_translation,
            orientation=self._euler_xyz_to_quat(cfg.wrist_camera_rpy),
        )
        self._left_camera = Camera(
            prim_path=f"{self._xarm_prim_path}/{cfg.left_camera_prim_name}",
            name=cfg.left_camera_name,
            resolution=cfg.left_camera_resolution,
            frequency=1.0 / cfg.rendering_dt,
        )
        self._left_camera.set_local_pose(
            translation=cfg.left_camera_translation,
            orientation=self._euler_xyz_to_quat(cfg.left_camera_rpy),
        )
        self._right_camera = Camera(
            prim_path=f"{self._xarm_prim_path}/{cfg.right_camera_prim_name}",
            name=cfg.right_camera_name,
            resolution=cfg.right_camera_resolution,
            frequency=1.0 / cfg.rendering_dt,
        )
        self._right_camera.set_local_pose(
            translation=cfg.right_camera_translation,
            orientation=self._euler_xyz_to_quat(cfg.right_camera_rpy),
        )

        if use_default_table:
            xarm_xform = UsdGeom.Xformable(self._world.stage.GetPrimAtPath(self._xarm_prim_path))
            translate_ops = [
                op for op in xarm_xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
            ]
            xarm_position = Gf.Vec3d(
                float(cfg.table_position[0]),
                float(cfg.table_position[1]),
                float(cfg.table_position[2] + cfg.table_scale[2] / 2.0),
            )
            if translate_ops:
                translate_ops[0].Set(xarm_position)
            else:
                xarm_xform.AddTranslateOp().Set(xarm_position)

        self._xarm = None
        if xarm_usd:
            self._xarm = Articulation(prim_path=self._xarm_prim_path, name="xarm")

        set_camera_view(eye=cfg.camera_eye, target=cfg.camera_target)

    @override
    def reset(self) -> None:
        self._world.reset()
        # Step once to ensure physics views are created before querying articulation state.
        self._world.step(render=False)
        self._step_count = 0
        self._last_action = np.zeros(7, dtype=np.float64)
        self._last_action_dict = self._default_action_dict()
        if self._xarm is not None:
            # Ensure articulation is initialized before sending targets.
            self._xarm.initialize()
        if self._wrist_camera is not None:
            self._wrist_camera.initialize()
        if self._left_camera is not None:
            self._left_camera.initialize()
        if self._right_camera is not None:
            self._right_camera.initialize()
        for _ in range(self.cfg.wrist_camera_warmup_steps):
            self._world.step(render=True)

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        return self._collect_observation()

    @override
    def apply_action(self, action: dict) -> None:
        if self._xarm is not None:
            targets = None
            use_velocity = False
            if "action_dict" in action:
                action_dict = action.get("action_dict") or {}
                self._last_action_dict = self._normalize_action_dict(action_dict)
                joint_position = self._last_action_dict.get("joint_position")
                joint_velocity = self._last_action_dict.get("joint_velocity")
                if joint_position is not None and np.asarray(joint_position).size:
                    targets = joint_position
                    self._last_action = np.asarray(joint_position, dtype=np.float64)
                    use_velocity = False
                elif joint_velocity is not None and np.asarray(joint_velocity).size:
                    targets = joint_velocity
                    self._last_action = np.asarray(joint_velocity, dtype=np.float64)
                    use_velocity = True
            elif "action" in action:
                self._last_action = np.asarray(action["action"], dtype=np.float64)
                targets = self._last_action
                self._last_action_dict = self._normalize_action_dict({"joint_position": targets})
            elif "actions" in action:
                targets = np.asarray(action["actions"], dtype=np.float64)
                self._last_action = targets
                self._last_action_dict = self._normalize_action_dict({"joint_position": targets})

            if targets is not None:
                targets = np.asarray(targets, dtype=np.float32)
                if targets.size:
                    if use_velocity and hasattr(self._xarm, "set_joint_velocity_targets"):
                        self._xarm.set_joint_velocity_targets(targets)
                    else:
                        if hasattr(self._xarm, "set_joint_position_targets"):
                            self._xarm.set_joint_position_targets(targets)
                        else:
                            self._xarm.set_joint_positions(targets)
        render = (not self.cfg.headless) or (self._wrist_camera is not None)
        self._world.step(render=render)
        self._step_count += 1

    def step(self, render: bool | None = None) -> None:
        if render is None:
            render = (not self.cfg.headless) or (self._wrist_camera is not None)
        self._world.step(render=render)

    def close(self) -> None:
        self._simulation_app.close()

    def get_joint_positions(self) -> np.ndarray | None:
        if self._xarm is None:
            return None
        qpos = self._xarm.get_joint_positions()
        if qpos is None:
            return None
        return np.asarray(qpos, dtype=np.float64)

    def _collect_observation(self) -> dict:
        state = self.get_joint_positions()
        images = {}
        if self._wrist_camera is not None:
            if hasattr(self._wrist_camera, "get_rgb"):
                rgb = self._wrist_camera.get_rgb()
                if rgb is not None:
                    images[self.cfg.wrist_image_left_key] = self._as_uint8(rgb)
            else:
                rgba = self._wrist_camera.get_rgba()
                if rgba is not None:
                    images[self.cfg.wrist_image_left_key] = self._as_uint8(rgba)[..., :3]
        if self._left_camera is not None:
            if hasattr(self._left_camera, "get_rgb"):
                rgb = self._left_camera.get_rgb()
                if rgb is not None:
                    images[self.cfg.exterior_image_1_left_key] = self._as_uint8(rgb)
            else:
                rgba = self._left_camera.get_rgba()
                if rgba is not None:
                    images[self.cfg.exterior_image_1_left_key] = self._as_uint8(rgba)[..., :3]
        if self._right_camera is not None:
            if hasattr(self._right_camera, "get_rgb"):
                rgb = self._right_camera.get_rgb()
                if rgb is not None:
                    images[self.cfg.exterior_image_2_left_key] = self._as_uint8(rgb)
            else:
                rgba = self._right_camera.get_rgba()
                if rgba is not None:
                    images[self.cfg.exterior_image_2_left_key] = self._as_uint8(rgba)[..., :3]

        joint_position = np.asarray(state if state is not None else np.zeros(7), dtype=np.float64)
        return {
            "gripper_position": np.zeros(1, dtype=np.float64),
            "cartesian_position": np.zeros(6, dtype=np.float64),
            "joint_position": joint_position,
            "wrist_image_left": images.get(self.cfg.wrist_image_left_key),
            "exterior_image_1_left": images.get(self.cfg.exterior_image_1_left_key),
            "exterior_image_2_left": images.get(self.cfg.exterior_image_2_left_key),
        }

    def _find_wrist_parent_path(self, usd_module, xarm_root: str, wrist_link_name: str) -> str:
        wrist_parent_path = f"{xarm_root}/{wrist_link_name}"
        if self._world.stage.GetPrimAtPath(wrist_parent_path).IsValid():
            return wrist_parent_path
        root_prim = self._world.stage.GetPrimAtPath(xarm_root)
        if root_prim.IsValid():
            for prim in usd_module.PrimRange(root_prim):
                if prim.GetName().lower() == wrist_link_name.lower():
                    found_path = prim.GetPath().pathString
                    return found_path
        print(
            f"Wrist link not found under {xarm_root}. "
            f"Attaching wrist camera to {xarm_root} instead."
        )
        return xarm_root

    @staticmethod
    def _resolve_asset_path(path: str, assets_root: str | None = None) -> str:
        if not path:
            return ""
        if path.startswith(("omniverse://", "http://", "https://")):
            return path
        if path.startswith("/"):
            return path
        if assets_root:
            return f"{assets_root.rstrip('/')}/{path}"
        return path

    @staticmethod
    def _join_assets_path(assets_root: str, relative_path: str) -> str:
        return f"{assets_root.rstrip('/')}/{relative_path.lstrip('/')}"

    @staticmethod
    def _as_uint8(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.dtype == np.uint8:
            return array
        if np.issubdtype(array.dtype, np.floating):
            max_val = float(np.nanmax(array)) if array.size else 1.0
            if max_val <= 1.0:
                array = array * 255.0
        return np.clip(array, 0, 255).astype(np.uint8)

    @staticmethod
    def _euler_xyz_to_quat(euler: np.ndarray) -> np.ndarray:
        quat_xyzw = R.from_euler("xyz", np.asarray(euler, dtype=np.float64)).as_quat()
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)

    @staticmethod
    def _default_action_dict() -> dict:
        return {
            "gripper_position": np.zeros(1, dtype=np.float64),
            "gripper_velocity": np.zeros(1, dtype=np.float64),
            "cartesian_position": np.zeros(6, dtype=np.float64),
            "cartesian_velocity": np.zeros(6, dtype=np.float64),
            "joint_position": np.zeros(7, dtype=np.float64),
            "joint_velocity": np.zeros(7, dtype=np.float64),
        }

    def _normalize_action_dict(self, action_dict: dict) -> dict:
        normalized = self._default_action_dict()
        for key, default in normalized.items():
            if key in action_dict and action_dict[key] is not None:
                normalized[key] = np.asarray(action_dict[key], dtype=np.float64)
            else:
                normalized[key] = np.asarray(default, dtype=np.float64)
        return normalized
