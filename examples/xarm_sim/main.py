import argparse
import logging
from collections import deque

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from env import EnvConfig, XArmIsaacEnvironment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isaac Sim XArm environment runner")
    parser.add_argument("--headless", action="store_true", help="Run without a viewer")
    parser.add_argument("--host", default="0.0.0.0", help="Policy server host")
    parser.add_argument("--port", type=int, default=8000, help="Policy server port")
    parser.add_argument("--prompt", default="do something", help="Prompt to send with observations")
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Resize images before policy inference",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EnvConfig(
        headless=args.headless,
    )
    env = XArmIsaacEnvironment(cfg)
    env.reset()

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port
    )
    logging.info("Server metadata: %s", policy.get_server_metadata())

    action_queue: deque[np.ndarray] = deque()

    try:
        while True:
            obs = env.get_observation()
            if _observation_missing_images(obs):
                env.step(render=not cfg.headless)
                continue

            if not action_queue:
                request = _build_policy_observation(obs, args.prompt, args.image_size)
                result = policy.infer(request)
                actions = np.asarray(result.get("actions"))
                if actions.ndim == 1:
                    action_queue.append(actions)
                else:
                    for action in actions:
                        action_queue.append(action)

            env.apply_action(_action_to_env(action_queue.popleft()))
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def _observation_missing_images(obs: dict) -> bool:
    return (
        obs.get("wrist_image_left") is None
        or obs.get("exterior_image_1_left") is None
        or obs.get("exterior_image_2_left") is None
    )


def _build_policy_observation(obs: dict, prompt: str, image_size: int) -> dict:
    return {
        "observation/wrist_image_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(obs["wrist_image_left"], image_size, image_size)
        ),
        "observation/exterior_image_1_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(obs["exterior_image_1_left"], image_size, image_size)
        ),
        "observation/exterior_image_2_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(obs["exterior_image_2_left"], image_size, image_size)
        ),
        "observation/joint_position": obs["joint_position"],
        "observation/gripper_position": obs["gripper_position"],
        "prompt": prompt,
    }


def _action_to_env(action: np.ndarray) -> dict:
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.size < 7:
        raise ValueError(f"Expected at least 7 action dims, got {action.size}")
    return {"actions": action[:7]}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
