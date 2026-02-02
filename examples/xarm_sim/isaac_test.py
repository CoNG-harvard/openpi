import argparse

import numpy as np
from isaacsim import SimulationApp


TABLE_POSITION = np.array([0.0, 0.0, 0.5])
TABLE_SCALE = np.array([1.5, 0.8, 0.05])
TABLE_COLOR = np.array([0.5, 0.5, 0.5])
CAMERA_EYE = [1.9542544578949033, 0.8892139220503694, 1.1613520627265954]
CAMERA_TARGET = [1.0874663592651088, 0.44882757271693136, 0.9273899816226002]
LIGHT_INTENSITY = 3000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load xArm7 USD in Isaac Sim")
    parser.add_argument("--headless", action="store_true", help="Run without a viewer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulation_app = SimulationApp({"headless": args.headless})

    from omni.isaac.core import World
    from omni.isaac.core.objects import FixedCuboid, GroundPlane
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.isaac.core.utils.prims import is_prim_path_valid
    from omni.isaac.core.utils.stage import add_reference_to_stage, set_stage_units
    from omni.isaac.core.utils.viewports import set_camera_view
    from pxr import Gf, UsdGeom, UsdLux

    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError("Isaac assets root is not available.")

    xarm_usd = f"{assets_root}/Isaac/Robots/Ufactory/xarm7/xarm7.usd"

    set_stage_units(1.0)
    world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    GroundPlane(prim_path="/World/GroundPlane", name="ground_plane")

    world.scene.add(
        FixedCuboid(
            prim_path="/World/Table",
            name="table",
            position=TABLE_POSITION,
            scale=TABLE_SCALE,
            color=TABLE_COLOR,
        )
    )

    light = UsdLux.DistantLight.Define(world.stage, "/World/KeyLight")
    light.CreateIntensityAttr(LIGHT_INTENSITY)

    add_reference_to_stage(xarm_usd, "/World/XArm")
    if not is_prim_path_valid("/World/XArm"):
        raise RuntimeError("Failed to load /World/XArm from the xArm USD.")
    set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)

    xarm_xform = UsdGeom.Xformable(world.stage.GetPrimAtPath("/World/XArm"))
    translate_ops = [
        op for op in xarm_xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
    ]
    xarm_position = Gf.Vec3d(
        float(TABLE_POSITION[0]),
        float(TABLE_POSITION[1]),
        float(TABLE_POSITION[2] + TABLE_SCALE[2] / 2.0),
    )
    if translate_ops:
        translate_ops[0].Set(xarm_position)
    else:
        xarm_xform.AddTranslateOp().Set(xarm_position)

    world.reset()

    if args.headless:
        simulation_app.close()
    else:
        while simulation_app.is_running():
            simulation_app.update()


if __name__ == "__main__":
    main()
