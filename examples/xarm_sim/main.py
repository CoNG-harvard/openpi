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
    parser.add_argument("--prompt", default="pick the red cube", help="Prompt to send with observations")
    parser.add_argument("--webrtc", dest="webrtc", action="store_true", help="Enable WebRTC streaming")
    parser.add_argument("--no-webrtc", dest="webrtc", action="store_false", help="Disable WebRTC streaming")
    parser.set_defaults(webrtc=True)
    parser.add_argument(
        "--open-loop-horizon",
        type=int,
        default=8,
        help="Number of actions to execute per policy inference",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EnvConfig(
        headless=args.headless,
        enable_webrtc=args.webrtc,
    )
    env = XArmIsaacEnvironment(cfg)
    env.reset()

    policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    action_queue: deque[np.ndarray] = deque()
    actions_from_chunk_completed = 0

    for _ in range(600):
        obs = env.get_observation()
        if (
            obs.get("wrist_image_left") is None
            or obs.get("exterior_image_1_left") is None
        ):
            env.step()
            continue

        if not action_queue or actions_from_chunk_completed >= args.open_loop_horizon:
            actions_from_chunk_completed = 0
            action_queue.clear()
            request = _build_policy_observation(obs, args.prompt)
            result = policy.infer(request)
            actions = np.asarray(result.get("actions"))
            if actions.ndim == 1:
                assert actions.shape[0] == 8, f"Expected 8 action dims, got {actions.shape}"
            elif actions.ndim == 2:
                assert actions.shape[1] == 8, f"Expected 8 action dims, got {actions.shape}"
            if actions.ndim == 1:
                action_queue.append(actions)
            else:
                for action in actions:
                    action_queue.append(action)

        action = np.asarray(action_queue.popleft(), dtype=np.float32).reshape(-1)
        if action.size:
            action[-1] = 1.0 if action[-1] > 0.5 else 0.0
        action = np.clip(action, -1.0, 1.0)
        env.apply_action({"actions": action})
        actions_from_chunk_completed += 1

    env.close()


def _build_policy_observation(obs: dict, prompt: str) -> dict:
    wrist = image_tools.resize_with_pad(obs["wrist_image_left"], 224, 224)
    exterior = image_tools.resize_with_pad(obs["exterior_image_1_left"], 224, 224)
    return {
        "observation/wrist_image_left": wrist,
        "observation/exterior_image_1_left": exterior,
        "observation/joint_position": obs["joint_position"],
        "observation/gripper_position": obs["gripper_position"],
        "prompt": prompt,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
