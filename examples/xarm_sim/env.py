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
    physics_dt: float = 1.0 / 60.0
    rendering_dt: float = 1.0 / 60.0
    stage_units_in_meters: float = 1.0
    xarm_usd_path: str = os.environ.get("XARM_USD_PATH", "")
    table_usd_path: str = os.environ.get("TABLE_USD_PATH", "")
    table_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.5]))
    table_scale: np.ndarray = field(default_factory=lambda: np.array([1.5, 0.8, 0.05]))
    table_color: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    camera_eye: list[float] = field(default_factory=lambda: [2.0, 0.9, 1.2])
    camera_target: list[float] = field(default_factory=lambda: [1.1, 0.4, 0.9])
    light_intensity: float = 4000.0
    wrist_camera_warmup_steps: int = 5
    camera_resolution: tuple[int, int] = (224, 224)
    wrist_camera_translation: np.ndarray = field(
        default_factory=lambda: np.array([0.00, 0.0, 0.02])
    )
    wrist_camera_rpy: np.ndarray = field(
        default_factory=lambda: np.array([0.0, -0.3, 3.14])
    )
    wrist_camera_name: str = "wrist_camera"
    wrist_camera_prim_name: str = "WristCamera"
    wrist_link_name: str = "link7"
    wrist_image_left_key: str = "wrist_image_left"
    left_camera_translation: np.ndarray = field(
        default_factory=lambda: np.array([0.4, 1.5, 0.8])
    )
    left_camera_rpy: np.ndarray = field(
        default_factory=lambda: np.array([-2.3, 0.0, -1.57])
    )
    left_camera_name: str = "left_camera"
    left_camera_prim_name: str = "LeftCamera"
    exterior_image_1_left_key: str = "exterior_image_1_left"
    
    # Front camera - looking at the workspace from the front
    front_camera_translation: np.ndarray = field(
        default_factory=lambda: np.array([1.8, 0.0, 0.8])
    )
    front_camera_rpy: np.ndarray = field(
        default_factory=lambda: np.array([-2.3, 0.0, 3.14])
    )
    front_camera_name: str = "front_camera"
    front_camera_prim_name: str = "FrontCamera"
    exterior_image_2_left_key: str = "exterior_image_2_left"
    
    # Overhead camera - top-down view
    overhead_camera_translation: np.ndarray = field(
        default_factory=lambda: np.array([0.3, 0.0, 2.5])
    )
    overhead_camera_rpy: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.57, 0])
    )
    overhead_camera_name: str = "overhead_camera"
    overhead_camera_prim_name: str = "OverheadCamera"
    exterior_image_3_left_key: str = "exterior_image_3_left"
    
    # Right camera - opposite side from left camera
    camera_resolution: tuple[int, int] = (224, 224)
    right_camera_translation: np.ndarray = field(
        default_factory=lambda: np.array([0.4, -1.5, 0.8])
    )
    right_camera_rpy: np.ndarray = field(
        default_factory=lambda: np.array([-2.3, 0.0, 1.57])
    )
    right_camera_name: str = "right_camera"
    right_camera_prim_name: str = "RightCamera"
    exterior_image_4_left_key: str = "exterior_image_4_left"
    ground_plane_prim_path: str = "/World/GroundPlane"
    ground_plane_name: str = "ground_plane"
    table_prim_path: str = "/World/Table"
    table_name: str = "table"
    cube_prim_path: str = "/World/TargetCube"
    cube_name: str = "target_cube"
    cube_position: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.0, 0.6]))
    cube_scale: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.2, 0.2]))
    cube_color: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    light_prim_path: str = "/World/KeyLight"
    
    # xArm 0
    xarm_0_prim_path: str = "/World/XArm_0"
    xarm_0_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.2, 0.55]))
    
    # xArm 1
    xarm_1_prim_path: str = "/World/XArm_1"
    xarm_1_position: np.ndarray = field(default_factory=lambda: np.array([0.0, -0.2, 0.55]))
    
    xarm_usd_default: str = "Isaac/Robots/Ufactory/xarm7/xarm7.usd"
    xarm_placeholder_size: float = 0.1
    xarm_placeholder_name: str = "xarm_placeholder"
    xarm_placeholder_child_name: str = "Placeholder"
    action_delta_scale: float = 0.05
    velocity_integration_dt: float = 0.1  # Typical for 10Hz policy loops
    # Initial joint positions: [j1, j2, j3, j4, j5, j6, j7] - arms raised with elbows bent
    initial_arm_positions: np.ndarray = field(
        default_factory=lambda: np.array([0.0, -0.5, 0.0, 0.8, 0.0, 0.5, 0.0])
    )
    initial_gripper_position: float = 0.0  # Gripper closed


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

        xarm_usd = cfg.xarm_usd_path or self._join_assets_path(assets_root, cfg.xarm_usd_default)
        xarm_usd = self._resolve_asset_path(xarm_usd, assets_root)
        if not xarm_usd:
            raise RuntimeError("XArm USD path is required.")

        # Initialize robot-specific data
        self._xarms: list[Articulation] = []
        self._wrist_cameras: list[Camera] = []
        self._wrist_prims: list[XFormPrim] = []
        
        for i, prim_path in enumerate([cfg.xarm_0_prim_path, cfg.xarm_1_prim_path]):
            add_reference_to_stage(xarm_usd, prim_path)
            if not is_prim_path_valid(prim_path):
                 raise RuntimeError(f"XArm {i} prim path missing after stage setup (usd={xarm_usd}).")
            
            # Set position
            xarm_xform = UsdGeom.Xformable(self._world.stage.GetPrimAtPath(prim_path))
            xarm_position = Gf.Vec3d(
                float([cfg.xarm_0_position, cfg.xarm_1_position][i][0]),
                float([cfg.xarm_0_position, cfg.xarm_1_position][i][1]),
                float([cfg.xarm_0_position, cfg.xarm_1_position][i][2]),
            )
            translate_ops = [
                op for op in xarm_xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
            ]
            if translate_ops:
                translate_ops[0].Set(xarm_position)
            else:
                xarm_xform.AddTranslateOp().Set(xarm_position)

            # Wrist Camera
            wrist_parent_path = self._find_wrist_parent_path(Usd, prim_path, cfg.wrist_link_name)
            self._wrist_prims.append(XFormPrim(prim_path=wrist_parent_path, name=f"{cfg.wrist_link_name}_{i}"))
            wrist_camera_path = f"{wrist_parent_path}/{cfg.wrist_camera_prim_name}_{i}"
            wrist_camera = Camera(
                prim_path=wrist_camera_path,
                name=f"{cfg.wrist_camera_name}_{i}",
                resolution=cfg.camera_resolution,
                frequency=1.0 / cfg.rendering_dt,
            )
            # Set local pose relative to parent link7
            wrist_camera.set_local_pose(
                translation=cfg.wrist_camera_translation,
                orientation=self._euler_xyz_to_quat(cfg.wrist_camera_rpy),
            )
            self._wrist_cameras.append(wrist_camera)
            self._xarms.append(Articulation(prim_path=prim_path, name=f"xarm_{i}"))

        # Left Camera
        self._left_camera = Camera(
            prim_path=f"/World/{cfg.left_camera_prim_name}",
            name=cfg.left_camera_name,
            position=cfg.left_camera_translation,
            orientation=self._euler_xyz_to_quat(cfg.left_camera_rpy),
            resolution=cfg.camera_resolution,
            frequency=1.0 / cfg.rendering_dt,
        )
        
        # Front Camera
        self._front_camera = Camera(
            prim_path=f"/World/{cfg.front_camera_prim_name}",
            name=cfg.front_camera_name,
            resolution=cfg.camera_resolution,
            position=cfg.front_camera_translation,
            orientation=self._euler_xyz_to_quat(cfg.front_camera_rpy),
            frequency=1.0 / cfg.rendering_dt,
        )
        
        # Overhead Camera
        self._overhead_camera = Camera(
            prim_path=f"/World/{cfg.overhead_camera_prim_name}",
            name=cfg.overhead_camera_name,
            resolution=cfg.camera_resolution,
            frequency=1.0 / cfg.rendering_dt,
            position=cfg.overhead_camera_translation,
            orientation=self._euler_xyz_to_quat(cfg.overhead_camera_rpy),
        )
        
        # Right Camera
        self._right_camera = Camera(
            prim_path=f"/World/{cfg.right_camera_prim_name}",
            name=cfg.right_camera_name,
            resolution=cfg.camera_resolution,
            position=cfg.right_camera_translation,
            orientation=self._euler_xyz_to_quat(cfg.right_camera_rpy),
            frequency=1.0 / cfg.rendering_dt,
        )

        set_camera_view(eye=cfg.camera_eye, target=cfg.camera_target)

    @override
    def reset(self) -> None:
        self._world.reset()
        # Step once to ensure physics views are created before querying articulation state.
        self._world.step(render=False)
        self._step_count = 0
        
        # Initialize all xArms
        self._dof_counts = []
        self._all_dof_names = []
        self._all_arm_indices = []
        self._all_gripper_indices = []

        for i, xarm in enumerate(self._xarms):
            xarm.initialize()
            dof_names = xarm.dof_names
            if dof_names:
                self._all_dof_names.append(list(dof_names))
                arm_idx, gripper_idx = self._infer_dof_groups(list(dof_names))
                self._all_arm_indices.append(arm_idx)
                self._all_gripper_indices.append(gripper_idx)
            
            current = xarm.get_joint_positions()
            if current is not None and current.size:
                self._dof_counts.append(int(current.size))
            else:
                self._dof_counts.append(13) # Default

        self._last_actions = [np.zeros(count, dtype=np.float64) for count in self._dof_counts]
        
        # Set initial joint positions to raise arms off the table
        for i, xarm in enumerate(self._xarms):
            initial_positions = np.zeros(self._dof_counts[i], dtype=np.float64)
            # Set arm joint positions
            arm_indices = self._all_arm_indices[i]
            for j, idx in enumerate(arm_indices):
                if j < len(self.cfg.initial_arm_positions):
                    initial_positions[idx] = self.cfg.initial_arm_positions[j]
            # Set gripper positions
            for idx in self._all_gripper_indices[i]:
                initial_positions[idx] = self.cfg.initial_gripper_position
            xarm.set_joint_positions(initial_positions)
        
        # Step to apply initial positions
        self._world.step(render=False)
        
        for cam in self._wrist_cameras:
            cam.initialize()
        if self._left_camera is not None:
            self._left_camera.initialize()
        if self._front_camera is not None:
            self._front_camera.initialize()
        if self._overhead_camera is not None:
            self._overhead_camera.initialize()
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
        # Expecting action["actions"] to be a list of actions for [robot0, robot1]
        # or a single concatenated array.
        targets_list = action.get("actions")
        if targets_list is None:
            return

        # Handle different action formats
        if isinstance(targets_list, (list, tuple)):
            actions = [np.asarray(t, dtype=np.float64).flatten() for t in targets_list]
        elif isinstance(targets_list, np.ndarray):
            # If it's a 2D array [batch, action_dim], we take the first chunk or split it
            if targets_list.ndim == 2:
                actions = [targets_list[i] for i in range(min(len(targets_list), len(self._xarms)))]
            else:
                # Concatenated? For now assume it's for robot 0 or needs splitting.
                # To be safe, let's assume if it's 1D, it's for Robot 0, unless it's long enough for both.
                total_expected = sum(len(self._all_arm_indices[i]) + 1 for i in range(len(self._xarms)))
                if targets_list.size == total_expected:
                    # Split it
                    actions = []
                    curr = 0
                    for i in range(len(self._xarms)):
                        size = len(self._all_arm_indices[i]) + 1
                        actions.append(targets_list[curr:curr+size])
                        curr += size
                else:
                    actions = [targets_list] # Just robot 0?
        else:
            actions = [np.asarray(targets_list, dtype=np.float64).flatten()]

        logging.debug(f"Applying actions to {len(actions)} robots")
        for i, (xarm, robot_action) in enumerate(zip(self._xarms, actions)):
            if i >= len(self._xarms):
                break
            
            logging.debug(f"Robot {i} receiving action shape: {robot_action.shape}, values: {robot_action[:3]}...")
            self._last_actions[i] = robot_action

            # Get current state for integration
            current_full = xarm.get_joint_positions()
            if current_full is None:
                current_full = np.zeros(self._dof_counts[i], dtype=np.float64)
                
            position_targets = current_full.copy()

            # Split targets based on indices
            arm_v, gripper_p = self._split_action_for_robot(robot_action, i)

            # Integrate Velocities for Arm
            if arm_v is not None:
                idx = np.array(self._all_arm_indices[i][:arm_v.size], dtype=np.int64)
                position_targets[idx] += arm_v[:idx.size] * self.cfg.velocity_integration_dt

            # Apply Absolute Positions for Gripper
            if gripper_p is not None:
                idx = np.array(self._all_gripper_indices[i], dtype=np.int64)
                if gripper_p.size == 1:
                    position_targets[idx] = gripper_p[0]
                else:
                    count = min(len(idx), gripper_p.size)
                    position_targets[idx[:count]] = gripper_p[:count]

            if hasattr(xarm, "set_joint_position_targets"):
                xarm.set_joint_position_targets(position_targets)
            else:
                xarm.set_joint_positions(position_targets)

        render = (not self.cfg.headless) or any(cam is not None for cam in self._wrist_cameras)
        self._world.step(render=render)
        self._step_count += 1

    def step(self, render: bool | None = None) -> None:
        if render is None:
            render = (not self.cfg.headless) or any(cam is not None for cam in self._wrist_cameras)
        self._world.step(render=render)
        
        if self._step_count % 60 == 0:
            try:
                cube = self._world.scene.get_object(self.cfg.cube_name)
                if cube:
                    pos, _ = cube.get_world_pose()
                
                if self._left_camera:
                     pos, orient = self._left_camera.get_world_pose()
                     # orient is usually w, x, y, z
                     
                if self._front_camera:
                     pos, _ = self._front_camera.get_world_pose()
            except Exception as e:
                pass

    def close(self) -> None:
        self._simulation_app.close()

    def get_joint_positions(self, robot_id: int = 0) -> np.ndarray | None:
        if robot_id >= len(self._xarms):
            return None
        xarm = self._xarms[robot_id]
        qpos = xarm.get_joint_positions()
        if qpos is None:
            return None
        return np.asarray(qpos, dtype=np.float64)

    def _collect_observation(self) -> dict:
        obs = {}
        for i, xarm in enumerate(self._xarms):
            state = self.get_joint_positions(i)
            joint_position = np.asarray(
                state if state is not None else np.zeros(self._dof_counts[i]),
                dtype=np.float64,
            )
            gripper_position = self._get_gripper_position(joint_position, i)
            
            obs[f"robot{i}_joint_position"] = joint_position
            obs[f"robot{i}_gripper_position"] = gripper_position
            
            if i < len(self._wrist_cameras):
                obs[f"robot{i}_wrist_image_left"] = self._wrist_cameras[i].get_rgb()
        
        if self._left_camera is not None:
            obs[self.cfg.exterior_image_1_left_key] = self._left_camera.get_rgb()
                
        if self._front_camera is not None:
            obs[self.cfg.exterior_image_2_left_key] = self._front_camera.get_rgb()
                
        if self._overhead_camera is not None:
            obs[self.cfg.exterior_image_3_left_key] = self._overhead_camera.get_rgb()
                
        if self._right_camera is not None:
            obs[self.cfg.exterior_image_4_left_key] = self._right_camera.get_rgb()
        
        # Maintain backward compatibility for robot 0
        obs["joint_position"] = obs.get("robot0_joint_position")
        obs["gripper_position"] = obs.get("robot0_gripper_position")
        obs["wrist_image_left"] = obs.get("robot0_wrist_image_left")
        
        return obs

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
    def _euler_xyz_to_quat(euler: np.ndarray) -> np.ndarray:
        quat_xyzw = R.from_euler("xyz", np.asarray(euler, dtype=np.float64)).as_quat()
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)

    def _default_action_dict(self) -> dict:
        # Note: This is now mostly for robot 0 or as a template
        count = self._dof_counts[0] if hasattr(self, "_dof_counts") and self._dof_counts else 13
        return {
            "gripper_position": np.zeros(1, dtype=np.float64),
            "gripper_velocity": np.zeros(1, dtype=np.float64),
            "cartesian_position": np.zeros(6, dtype=np.float64),
            "cartesian_velocity": np.zeros(6, dtype=np.float64),
            "joint_position": np.zeros(count, dtype=np.float64),
            "joint_velocity": np.zeros(count, dtype=np.float64),
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

    def _split_action_for_robot(self, action: np.ndarray, robot_id: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        action = np.asarray(action, dtype=np.float64).flatten()
        if not action.size:
            return None, None
        
        arm_count = len(self._all_arm_indices[robot_id])
        
        # Priority 1: Exact match with arm + gripper_count
        if action.size == arm_count + len(self._all_gripper_indices[robot_id]):
            return action[:arm_count], action[arm_count:]
            
        # Priority 2: Standard DROID match [arm + 1]
        if action.size == arm_count + 1:
            return action[:arm_count], action[arm_count:]
            
        # Fallback: Treat as much as possible as arm, rest as gripper
        if action.size >= arm_count:
             return action[:arm_count], action[arm_count:] if action.size > arm_count else None
             
        return action, None

    def _expand_gripper_targets(self, gripper_command: np.ndarray, current_full: np.ndarray, robot_id: int) -> np.ndarray | None:
        if not self._all_gripper_indices[robot_id]:
            return None
        gripper_command = np.asarray(gripper_command, dtype=np.float32).reshape(-1)
        if not gripper_command.size:
            return None
        targets = current_full.copy()
        indices = np.array(self._all_gripper_indices[robot_id], dtype=np.int64)
        if gripper_command.size == 1:
            targets[indices] = gripper_command[0]
        else:
            count = min(len(indices), gripper_command.size)
            targets[indices[:count]] = gripper_command[:count]
        return targets

    def _set_joint_position_targets(self, targets: np.ndarray, robot_id: int = 0) -> None:
        if robot_id >= len(self._xarms):
            return
        xarm = self._xarms[robot_id]
        if hasattr(xarm, "set_joint_position_targets"):
            xarm.set_joint_position_targets(targets)
        else:
            xarm.set_joint_positions(targets)

    def _get_gripper_position(self, joint_position: np.ndarray, robot_id: int = 0) -> np.ndarray:
        if robot_id >= len(self._all_gripper_indices):
            return np.zeros(1, dtype=np.float64)
        indices = np.array(self._all_gripper_indices[robot_id], dtype=np.int64)
        if joint_position.size <= indices.max():
            return np.zeros(1, dtype=np.float64)
        return np.array([float(np.mean(joint_position[indices]))], dtype=np.float64)

    def _get_wrist_cartesian_position(self, robot_id: int = 0) -> np.ndarray:
        if robot_id >= len(self._wrist_prims):
            return np.zeros(6, dtype=np.float64)
        wrist_prim = self._wrist_prims[robot_id]
        try:
            position, orientation = wrist_prim.get_world_pose()
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
