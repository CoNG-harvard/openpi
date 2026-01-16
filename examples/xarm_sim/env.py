import logging
import os
from dataclasses import dataclass, field

import numpy as np
from openpi_client.runtime import environment as _environment
from scipy.spatial.transform import Rotation as R
from typing_extensions import override


@dataclass
class EnvConfig:
    headless: bool = True
    enable_webrtc: bool = True
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
    wrist_camera_resolution: tuple[int, int] = (224, 224)
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
    left_camera_resolution: tuple[int, int] = (224, 224)
    left_camera_translation: np.ndarray = field(
        default_factory=lambda: np.array([0.4155, 0.4775, 0.4114])
    )
    left_camera_rpy: np.ndarray = field(
        default_factory=lambda: np.array([-2.2160, -0.0226, -2.9637])
    )
    left_camera_name: str = "left_camera"
    left_camera_prim_name: str = "LeftCamera"
    exterior_image_1_left_key: str = "exterior_image_1_left"
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
    action_delta_scale: float = 0.05
    velocity_integration_dt: float = 0.1  # Typical for 10Hz policy loops


class XArmIsaacEnvironment(_environment.Environment):
    """A minimal Isaac Sim environment for an xArm pick task."""

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self._dof_count = 13
        self._dof_names: list[str] = []
        self._arm_dof_indices: list[int] = list(range(7))
        self._gripper_dof_indices: list[int] = list(range(7, 13))
        self._step_count = 0
        self._last_action = np.zeros(self._dof_count, dtype=np.float64)
        self._last_action_dict = self._default_action_dict()
        from isaacsim import SimulationApp

        launch_config = {"headless": cfg.headless}
        if cfg.enable_webrtc:
            # hide_ui=False is required for the video capture process to function correctly.
            launch_config["hide_ui"] = False

        self._simulation_app = SimulationApp(launch_config)

        if cfg.enable_webrtc:
            import omni.kit.app
            omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.kit.livestream.webrtc", True)

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
        if not xarm_usd:
            raise RuntimeError("XArm USD path is required.")
        add_reference_to_stage(xarm_usd, self._xarm_prim_path)

        if not is_prim_path_valid(self._xarm_prim_path):
            raise RuntimeError(f"XArm prim path missing after stage setup (usd={xarm_usd}).")

        self._wrist_camera = None
        self._wrist_prim = None
        self._left_camera = None
        wrist_parent_path = self._find_wrist_parent_path(Usd, self._xarm_prim_path, cfg.wrist_link_name)
        self._wrist_prim = XFormPrim(prim_path=wrist_parent_path, name=cfg.wrist_link_name)
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
        if self._xarm is not None:
            # Ensure articulation is initialized before sending targets.
            self._xarm.initialize()
            dof_names = self._xarm.dof_names
            if dof_names:
                self._dof_names = list(dof_names)
                self._arm_dof_indices, self._gripper_dof_indices = self._infer_dof_groups(self._dof_names)
                logging.info("xArm DOF names: %s", list(enumerate(dof_names)))
                logging.info(
                    "xArm DOF groups: arm=%s gripper=%s",
                    self._arm_dof_indices,
                    self._gripper_dof_indices,
                )
            current = self.get_joint_positions()
            if current is not None and current.size:
                self._dof_count = int(current.size)
        self._last_action = np.zeros(self._dof_count, dtype=np.float64)
        self._last_action_dict = self._default_action_dict()
        if self._wrist_camera is not None:
            self._wrist_camera.initialize()
        if self._left_camera is not None:
            self._left_camera.initialize()
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
        if self._xarm is None:
            return

        # 1. Extract raw targets
        targets = action.get("actions")
        if targets is None and "action_dict" in action:
            targets = action["action_dict"].get("joint_position")
        
        if targets is None:
            return
            
        targets = np.asarray(targets, dtype=np.float64).flatten()
        self._last_action = targets

        # 2. Get current state for integration
        current_full = self.get_joint_positions()
        if current_full is None:
            current_full = np.zeros(self._dof_count, dtype=np.float64)
            
        position_targets = current_full.copy()

        # 3. Split targets based on indices
        arm_v, gripper_p = self._split_action(targets)

        # 4. Integrate Velocities for Arm
        if arm_v is not None:
            # Match provided velocities to arm joints
            idx = np.array(self._arm_dof_indices[:arm_v.size], dtype=np.int64)
            position_targets[idx] += arm_v[:idx.size] * self.cfg.velocity_integration_dt

        # 5. Apply Absolute Positions for Gripper
        if gripper_p is not None:
            idx = np.array(self._gripper_dof_indices, dtype=np.int64)
            if gripper_p.size == 1:
                # DROID case: broadcast 1 scalar to all gripper DOFs
                position_targets[idx] = gripper_p[0]
            else:
                # Multi-DOF gripper case: use as many as provided
                count = min(len(idx), gripper_p.size)
                position_targets[idx[:count]] = gripper_p[:count]

        self._set_joint_position_targets(position_targets)
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
            rgb = self._wrist_camera.get_rgb()
            if rgb is not None:
                images[self.cfg.wrist_image_left_key] = self._as_uint8(rgb)
        if self._left_camera is not None:
            rgb = self._left_camera.get_rgb()
            if rgb is not None:
                images[self.cfg.exterior_image_1_left_key] = self._as_uint8(rgb)
        joint_position = np.asarray(
            state if state is not None else np.zeros(self._dof_count),
            dtype=np.float64,
        )
        gripper_position = self._get_gripper_position(joint_position)
        return {
            "gripper_position": gripper_position,
            "joint_position": joint_position,
            "wrist_image_left": images.get(self.cfg.wrist_image_left_key),
            "exterior_image_1_left": images.get(self.cfg.exterior_image_1_left_key),
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

    def _default_action_dict(self) -> dict:
        return {
            "gripper_position": np.zeros(1, dtype=np.float64),
            "gripper_velocity": np.zeros(1, dtype=np.float64),
            "cartesian_position": np.zeros(6, dtype=np.float64),
            "cartesian_velocity": np.zeros(6, dtype=np.float64),
            "joint_position": np.zeros(self._dof_count, dtype=np.float64),
            "joint_velocity": np.zeros(self._dof_count, dtype=np.float64),
        }

    def _normalize_action_dict(self, action_dict: dict) -> dict:
        normalized = self._default_action_dict()
        for key, default in normalized.items():
            if key in action_dict and action_dict[key] is not None:
                normalized[key] = np.asarray(action_dict[key], dtype=np.float64)
            else:
                normalized[key] = np.asarray(default, dtype=np.float64)
        return normalized

    def _infer_dof_groups(self, dof_names: list[str]) -> tuple[list[int], list[int]]:
        if not dof_names:
            return [], []
        lower = [name.lower() for name in dof_names]
        gripper_keywords = ("gripper", "finger", "knuckle", "pad", "tip", "mimic", "drive")
        gripper_indices = [i for i, name in enumerate(lower) if any(k in name for k in gripper_keywords)]
        arm_indices = [i for i in range(len(lower)) if i not in gripper_indices]
        return arm_indices, gripper_indices

    def _split_action(self, action: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        action = np.asarray(action, dtype=np.float64).flatten()
        if not action.size:
            return None, None
        
        arm_count = len(self._arm_dof_indices)
        
        # Priority 1: Exact match with arm + gripper_count
        if action.size == arm_count + len(self._gripper_dof_indices):
            return action[:arm_count], action[arm_count:]
            
        # Priority 2: Standard DROID match [arm + 1]
        if action.size == arm_count + 1:
            return action[:arm_count], action[arm_count:]
            
        # Fallback: Treat as much as possible as arm, rest as gripper
        if action.size >= arm_count:
             return action[:arm_count], action[arm_count:] if action.size > arm_count else None
             
        return action, None

    def _expand_gripper_targets(self, gripper_command: np.ndarray, current_full: np.ndarray) -> np.ndarray | None:
        if not self._gripper_dof_indices:
            return None
        gripper_command = np.asarray(gripper_command, dtype=np.float32).reshape(-1)
        if not gripper_command.size:
            return None
        targets = current_full.copy()
        indices = np.array(self._gripper_dof_indices, dtype=np.int64)
        if gripper_command.size == 1:
            targets[indices] = gripper_command[0]
        else:
            count = min(len(indices), gripper_command.size)
            targets[indices[:count]] = gripper_command[:count]
        return targets

    def _set_joint_position_targets(self, targets: np.ndarray) -> None:
        if hasattr(self._xarm, "set_joint_position_targets"):
            self._xarm.set_joint_position_targets(targets)
        else:
            self._xarm.set_joint_positions(targets)

    def _get_gripper_position(self, joint_position: np.ndarray) -> np.ndarray:
        if not self._gripper_dof_indices:
            return np.zeros(1, dtype=np.float64)
        indices = np.array(self._gripper_dof_indices, dtype=np.int64)
        if joint_position.size <= indices.max():
            return np.zeros(1, dtype=np.float64)
        return np.array([float(np.mean(joint_position[indices]))], dtype=np.float64)

    def _get_wrist_cartesian_position(self) -> np.ndarray:
        if self._wrist_prim is None:
            return np.zeros(6, dtype=np.float64)
        try:
            position, orientation = self._wrist_prim.get_world_pose()
        except Exception:
            return np.zeros(6, dtype=np.float64)
        position = np.asarray(position, dtype=np.float64).reshape(-1)
        orientation = np.asarray(orientation, dtype=np.float64).reshape(-1)
        if position.size < 3 or orientation.size < 4:
            return np.zeros(6, dtype=np.float64)
        euler = self._quat_wxyz_to_euler_xyz(orientation[:4])
        return np.concatenate([position[:3], euler]).astype(np.float64)

    @staticmethod
    def _quat_wxyz_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float64).reshape(-1)
        if quat.size != 4:
            return np.zeros(3, dtype=np.float64)
        quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)
        return R.from_quat(quat_xyzw).as_euler("xyz")
