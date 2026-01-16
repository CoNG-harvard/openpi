import argparse
import logging

import cv2
import numpy as np
from collections import deque

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
    parser.add_argument(
        "--random-policy",
        action="store_true",
        help="Use a random oscillating policy instead of the policy server",
    )
    return parser.parse_args()


class OscillatingPolicy:
    def __init__(self, horizon: int = 8):
        self.horizon = horizon
        self.direction = np.ones(7)
        self.joint_step = 0.05
        self.joint_target_limit = 1.0
        self.gripper_val = 1.0

    def infer(self, request: dict) -> dict:
        current_joints = np.array(request["observation/joint_position"][:7])
        actions = []
        state = current_joints.copy()
        
        for _ in range(self.horizon):
            next_targets = state + self.direction * self.joint_step
            if (next_targets > self.joint_target_limit).any() or (next_targets < -self.joint_target_limit).any():
                self.direction *= -1.0
                next_targets = state + self.direction * self.joint_step
            
            state = next_targets
            action = np.concatenate([state, [self.gripper_val]])
            actions.append(action)
            
        return {"actions": np.array(actions)}


def main() -> None:
    args = parse_args()
    
    cfg = EnvConfig(
        headless=args.headless,
        enable_webrtc=args.webrtc,
    )
    
    env = XArmIsaacEnvironment(cfg)
    env.reset()

    if args.random_policy:
        logging.info("Using OscillatingPolicy")
        policy = OscillatingPolicy(horizon=args.open_loop_horizon)
    else:
        logging.info(f"Connecting to policy server at {args.host}:{args.port}")
        policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    action_queue: deque[np.ndarray] = deque()
    actions_from_chunk_completed = 0
    
    # Create a single window for both cameras
    cv2.namedWindow("Robot Cameras", cv2.WINDOW_NORMAL)

    try:
        while True:
            # Ensure rendering happens before capturing images
            env.step(render=True)
            
            obs = env.get_observation()
            if (
                obs.get("wrist_image_left") is None
                or obs.get("exterior_image_1_left") is None
            ):
                continue
            
            # Get images
            wrist_img = obs["wrist_image_left"]
            exterior_img = obs["exterior_image_1_left"]
            
            # Convert RGB to BGR for OpenCV display
            wrist_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
            exterior_bgr = cv2.cvtColor(exterior_img, cv2.COLOR_RGB2BGR)
            
            # Concatenate images horizontally (side by side)
            combined = np.hstack([wrist_bgr, exterior_bgr])
            
            # Add labels
            cv2.putText(combined, "Wrist Camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, "Exterior Camera", (wrist_img.shape[1] + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Robot Cameras", combined)
            cv2.waitKey(1)  # Refresh display

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
            
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received, stopping simulation.")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
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
