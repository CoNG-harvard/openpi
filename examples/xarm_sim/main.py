import argparse

from env import EnvConfig, XArmIsaacEnvironment


def parse_args() -> argparse.Namespace:
    defaults = EnvConfig()
    parser = argparse.ArgumentParser(description="Isaac Sim XArm environment runner")
    parser.add_argument("--headless", action="store_true", help="Run without a viewer")
    parser.add_argument(
        "--joint-step",
        type=float,
        default=defaults.joint_step,
        help="Increment per step for joint targets",
    )
    parser.add_argument(
        "--joint-limit",
        type=float,
        default=defaults.joint_target_limit,
        help="Absolute joint target limit",
    )
    parser.add_argument(
        "--initial-direction",
        type=float,
        default=defaults.initial_direction,
        help="Initial joint target direction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EnvConfig(
        headless=args.headless,
        joint_step=args.joint_step,
        joint_target_limit=args.joint_limit,
        initial_direction=args.initial_direction,
    )
    env = XArmIsaacEnvironment(cfg)
    env.reset()

    direction = cfg.initial_direction
    try:
        while True:
            obs = env.get_observation()
            print(f"observation keys: {list(obs.keys())}")
            state = env.get_joint_positions()
            if state is None:
                env.step(render=not cfg.headless)
                continue

            next_targets = state + direction * cfg.joint_step
            if (next_targets > cfg.joint_target_limit).any() or (next_targets < -cfg.joint_target_limit).any():
                direction *= -1.0
                next_targets = state + direction * cfg.joint_step
            env.apply_action({"actions": next_targets})
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
