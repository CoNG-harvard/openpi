import argparse
import logging
import os

import cv2
import numpy as np

from openpi_client import websocket_client_policy as _websocket_client_policy

from env import EnvConfig, XArmIsaacEnvironment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isaac Sim XArm environment runner")
    parser.add_argument("--headless", action="store_true", help="Run without a viewer")
    parser.add_argument("--host", default="0.0.0.0", help="Policy server host")
    parser.add_argument("--port", type=int, default=8000, help="Policy server port")
    parser.add_argument("--prompt", default="pick the cube", help="Prompt to send with observations")
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use a random oscillating policy instead of the policy server",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="How many actions to execute from a predicted action chunk before querying policy server again",
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
    )
    
    env = XArmIsaacEnvironment(cfg)
    env.reset()

    if args.random:
        logging.info("Using OscillatingPolicy")
        policy = OscillatingPolicy(num_robots=2)
    else:
        logging.info(f"Connecting to policy server at {args.host}:{args.port}")
        policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    if not args.headless:
        # Create a single window for both cameras
        cv2.namedWindow("Robot Cameras", cv2.WINDOW_NORMAL)

    # Rollout parameters
    actions_from_chunk_completed = 0
    pred_action_chunk = None
    snapshot_saved = False

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
            robot0_wrist = obs.get("robot0_wrist_image_left")
            robot1_wrist = obs.get("robot1_wrist_image_left")
            exterior_1 = obs.get("exterior_image_1_left")
            exterior_2 = obs.get("exterior_image_2_left")
            exterior_3 = obs.get("exterior_image_3_left")
            exterior_4 = obs.get("exterior_image_4_left")
            
            if not args.headless:
                images_to_show = []
                # Helper to process image
                def process_img(img, label):
                    if img is None:
                        # Placeholder if missing
                        img = np.zeros((224, 224, 3), dtype=np.uint8)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    # Add label
                    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    return img

                # Row 1: Wrist Cameras
                row1 = np.hstack([
                    process_img(robot0_wrist, "Robot 0 Wrist"),
                    process_img(robot1_wrist, "Robot 1 Wrist")
                ])
                
                # Row 2: Fixed Cameras 1 & 2
                row2 = np.hstack([
                    process_img(exterior_1, "Left Camera"),
                    process_img(exterior_2, "Front Camera")
                ])

                # Row 3: Fixed Cameras 3 & 4
                row3 = np.hstack([
                    process_img(exterior_3, "Overhead Camera"),
                    process_img(exterior_4, "Right Camera")
                ])
                
                # Stack all rows
                combined = np.vstack([row1, row2, row3])
                
                cv2.imshow("Robot Cameras", combined)
                cv2.waitKey(1)  # Refresh display

                # SNAPSHOT for debugging
                if not snapshot_saved and os.path.exists("/home/lening/.gemini/antigravity/brain/1bb64de2-d19b-4323-9477-2257f9c4f16e"):
                     cv2.imwrite("/home/lening/.gemini/antigravity/brain/1bb64de2-d19b-4323-9477-2257f9c4f16e/snapshot.png", combined)
                     snapshot_saved = True
                     print("DEBUG: Snapshot saved to artifacts.")


            if args.random:
                request = _build_policy_observation(obs, args.prompt)
                result = policy.infer(request)
                action = result.get("actions") # List of arrays
            else:
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.horizon:
                    actions_from_chunk_completed = 0
                    request = _build_policy_observation(obs, args.prompt)
                    # this usually returns action chunk [horizon, action_dim]
                    pred_action_chunk = np.asarray(policy.infer(request)["actions"])
                
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

            env.apply_action({"actions": action})
            
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
        "observation/exterior_image_2_left": obs.get("exterior_image_2_left"),
        "observation/exterior_image_3_left": obs.get("exterior_image_3_left"),
        "observation/exterior_image_4_left": obs.get("exterior_image_4_left"),
        # Backward compatibility for robot 0
        "observation/wrist_image_left": obs.get("wrist_image_left"),
        "observation/joint_position": obs.get("joint_position"),
        "observation/gripper_position": obs.get("gripper_position"),
        "prompt": prompt,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
