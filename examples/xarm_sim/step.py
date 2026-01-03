import argparse
import random

from env import EnvConfig, XArmIsaacEnvironment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step Isaac Sim XArm environment without policy inference")
    parser.add_argument("--headless", action="store_true", help="Run without a viewer")
    parser.add_argument("--xarm-usd", default="", help="Path to xArm USD file")
    parser.add_argument("--table-usd", default="", help="Path to a table USD file")
    parser.add_argument("--steps", type=int, default=600, help="Number of sim steps to run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EnvConfig(
        headless=args.headless,
        xarm_usd_path=args.xarm_usd,
        table_usd_path=args.table_usd,
    )
    env = XArmIsaacEnvironment(cfg)
    env.reset()

    for i in range(args.steps):
        action = [random.uniform(-1.0, 1.0) for _ in range(7)]
        print(f"Step {i + 1}/{args.steps} action={action}")
        env.apply_action({"actions": action})
        print(f"observation: {env.get_observation()}")

    env.close()


if __name__ == "__main__":
    main()
