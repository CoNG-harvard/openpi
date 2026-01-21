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
        "--random-policy",
        action="store_true",
        help="Use a random oscillating policy instead of the policy server",
    )
    return parser.parse_args()


class OscillatingPolicy:
    def __init__(self, num_robots: int = 2):
        self.num_robots = num_robots
        self.directions = [np.ones(7) for _ in range(num_robots)]
        self.joint_step = 0.05
        self.joint_target_limit = 1.0
        self.gripper_val = 1.0

    def infer(self, request: dict) -> dict:
        actions = []
        for i in range(self.num_robots):
            joint_key = f"observation/robot{i}_joint_position"
            if joint_key not in request:
                # Fallback to robot 0 if only one set is present
                joint_key = "observation/joint_position"
            
            current_joints = np.array(request[joint_key][:7])
            state = current_joints.copy()
            
            # Check limits
            next_pos = state + self.directions[i] * self.joint_step
            if (next_pos > self.joint_target_limit).any() or (next_pos < -self.joint_target_limit).any():
                self.directions[i] *= -1.0
            
            velocity = self.directions[i] * self.joint_step
            # Action consists of [v1, v2, ..., v7, gripper_pos]
            action = np.concatenate([velocity, [self.gripper_val]])
            actions.append(action)
            
        return {"actions": actions}


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
        policy = OscillatingPolicy(num_robots=2)
    else:
        logging.info(f"Connecting to policy server at {args.host}:{args.port}")
        policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    if not args.headless:
        # Create a single window for both cameras
        cv2.namedWindow("Robot Cameras", cv2.WINDOW_NORMAL)

    try:
        while True:
            # Ensure rendering happens before capturing images
            env.step(render=True)
            
            obs = env.get_observation()
            if (
                obs.get("robot0_wrist_image_left") is None
                or obs.get("robot1_wrist_image_left") is None
                or obs.get("exterior_image_1_left") is None
            ):
                continue
            
            # Get images
            robot0_wrist = obs["robot0_wrist_image_left"]
            robot1_wrist = obs["robot1_wrist_image_left"]
            exterior_img = obs["exterior_image_1_left"]
            
            if not args.headless:
                # Convert RGB to BGR for OpenCV display
                r0_wrist_bgr = cv2.cvtColor(robot0_wrist, cv2.COLOR_RGB2BGR)
                r1_wrist_bgr = cv2.cvtColor(robot1_wrist, cv2.COLOR_RGB2BGR)
                exterior_bgr = cv2.cvtColor(exterior_img, cv2.COLOR_RGB2BGR)
                
                # Resize exterior to match wrist images height if needed, 
                # or just stack them. Let's stack them in a grid.
                top_row = np.hstack([r0_wrist_bgr, r1_wrist_bgr])
                
                # Resize exterior to match the width of top_row
                ext_resized = cv2.resize(exterior_bgr, (top_row.shape[1], int(exterior_bgr.shape[0] * top_row.shape[1] / exterior_bgr.shape[1])))
                
                combined = np.vstack([top_row, ext_resized])
                
                # Add labels
                cv2.putText(combined, "Robot 0 Wrist", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(combined, "Robot 1 Wrist", (robot0_wrist.shape[1] + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(combined, "Exterior Camera", (10, top_row.shape[0] + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Robot Cameras", combined)
                cv2.waitKey(1)  # Refresh display

            request = _build_policy_observation(obs, args.prompt)
            result = policy.infer(request)
            actions = result.get("actions")
            
            # actions is expected to be a list of arrays [r0_action, r1_action]
            env.apply_action({"actions": actions})
            
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received, stopping simulation.")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        if not args.headless:
            cv2.destroyAllWindows()
        env.close()


def _build_policy_observation(obs: dict, prompt: str) -> dict:
    return {
        "observation/robot0_wrist_image_left": obs.get("robot0_wrist_image_left"),
        "observation/robot1_wrist_image_left": obs.get("robot1_wrist_image_left"),
        "observation/robot0_joint_position": obs.get("robot0_joint_position"),
        "observation/robot1_joint_position": obs.get("robot1_joint_position"),
        "observation/robot0_gripper_position": obs.get("robot0_gripper_position"),
        "observation/robot1_gripper_position": obs.get("robot1_gripper_position"),
        "observation/exterior_image_1_left": obs.get("exterior_image_1_left"),
        # Backward compatibility for robot 0
        "observation/wrist_image_left": obs.get("wrist_image_left"),
        "observation/joint_position": obs.get("joint_position"),
        "observation/gripper_position": obs.get("gripper_position"),
        "prompt": prompt,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
