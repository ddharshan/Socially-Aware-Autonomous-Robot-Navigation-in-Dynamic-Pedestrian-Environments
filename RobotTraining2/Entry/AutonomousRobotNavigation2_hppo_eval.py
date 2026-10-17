import os
import time
import gym
import torch
import numpy as np
from tensorboardX import SummaryWriter

from ding.config import compile_config
from ding.worker import (
    BaseLearner,
    SampleSerialCollector,
    InteractionSerialEvaluator,
    AdvancedReplayBuffer,
)
from ding.envs import BaseEnvManager
from ding.policy import PPOPolicy
from ding.model import VAC
from ding.utils import set_pkg_seed

from RobotTraining2.Config.AutonomousRobotNavigation2_hppo_config import (
    AutonomousRobotNavigation2_hppo_config,
    AutonomousRobotNavigation2_hppo_create_config,
)

# This import is important so gym can register DRLRobot-v0
import AutonomousRobotNavigation2  # noqa: F401


# ============================================================
# SETTINGS FOR 12-EPISODE TESTING ONLY
# ============================================================

N_EVAL_EPISODES = 12
SAVE_VIDEO = True
VIDEO_FRAME_SKIP = 1
VIDEO_FPS = 30

CKPT_PATH = (
    r"D:\MSc Robot Navigation\RLRobotTraining2"
    r"\AutonomousRobotNavigation2_hppo_seed0_retrain_from_best71_metrics"
    r"\ckpt\ckpt_best.pth.tar"
)

OUTPUT_ROOT = (
    r"D:\MSc Robot Navigation\RLRobotTraining2"
    r"\AutonomousRobotNavigation2_hppo_seed0_retrain_from_best71_metrics"
)


try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def extract_hybrid_action(policy_output):
    """
    Convert DI-engine hybrid PPO action output into your raw Gym action:
        (action_id, action_parameters)

    Your action space:
        action_id: 0 accelerate, 1 turn, 2 break
        action_parameters: [acceleration_param, turn_param]
    """
    if isinstance(policy_output, dict):
        if 0 in policy_output:
            out = policy_output[0]
        else:
            out = list(policy_output.values())[0]
    else:
        out = policy_output

    action = out["action"]

    if isinstance(action, dict):
        action_type = action.get("action_type", None)
        action_args = action.get("action_args", None)

        if action_type is None:
            action_type = action.get("type", None)
        if action_args is None:
            action_args = action.get("args", None)

        action_id = int(to_numpy(action_type).reshape(-1)[0])
        action_params = to_numpy(action_args).astype(np.float32).reshape(-1)

    elif isinstance(action, (tuple, list)):
        action_id = int(to_numpy(action[0]).reshape(-1)[0])
        action_params = to_numpy(action[1]).astype(np.float32).reshape(-1)

    else:
        raise TypeError(f"Unsupported action format: {type(action)}")

    action_id = int(np.clip(action_id, 0, 2))

    if action_params.size < 2:
        padded = np.zeros(2, dtype=np.float32)
        padded[:action_params.size] = action_params
        action_params = padded

    action_params = action_params[:2].astype(np.float32)
    action_params[0] = np.clip(action_params[0], 0.0, 1.0)
    action_params[1] = np.clip(action_params[1], -1.0, 1.0)

    return action_id, action_params


def nearest_pedestrian_distance(env):
    return min(
        env.collision1,
        env.collision2,
        env.collision3,
        env.collision4,
        env.collision5,
        env.collision6,
    )


def infer_done_reason(env, info):
    """
    Uses info['done_reason'] from the modified environment.py.
    If info is missing, it infers the reason manually.
    """
    if isinstance(info, dict) and "done_reason" in info:
        return info["done_reason"]

    if env.distance <= env.target_radius:
        return "goal"

    if abs(env.agent.x) > env.field_size or abs(env.agent.y) > env.field_size:
        return "boundary"

    if env.current_step > env.max_step:
        return "timeout"

    if nearest_pedestrian_distance(env) <= 0.21:
        return "collision"

    return "unknown"


def mean(values):
    values = [v for v in values if v is not None]
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def get_info_value(info, key, default=None):
    if isinstance(info, dict) and key in info:
        return info[key]
    return default


def main(main_cfg, create_cfg, seed=0):
    # Random seed every run, so setup changes each time.
    seed = int(time.time() * 1000) % 1000000

    # Force clean 12-episode testing.
    main_cfg.env.evaluator_env_num = 1
    main_cfg.env.n_evaluator_episode = N_EVAL_EPISODES

    cfg = compile_config(
        main_cfg,
        BaseEnvManager,
        PPOPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        AdvancedReplayBuffer,
        create_cfg=create_cfg,
        save_cfg=True,
    )

    if not create_cfg.policy.type.endswith("_command"):
        create_cfg.policy.type = create_cfg.policy.type + "_command"

    cfg = compile_config(
        cfg,
        seed=seed,
        env=None,
        auto=True,
        create_cfg=create_cfg,
        save_cfg=True,
    )

    cfg.env.evaluator_env_num = 1
    cfg.env.n_evaluator_episode = N_EVAL_EPISODES

    assert os.path.exists(CKPT_PATH), f"Checkpoint not found: {CKPT_PATH}"

    replay_dir = os.path.join(
        OUTPUT_ROOT,
        f"video_12ep_metrics_seed_{seed}"
    )
    os.makedirs(replay_dir, exist_ok=True)

    print("============================================")
    print("12-EPISODE METRIC TESTING")
    print("Random eval seed:", seed)
    print("Episodes:", N_EVAL_EPISODES)
    print("Checkpoint:", CKPT_PATH)
    print("Video folder:", replay_dir)
    print("============================================")

    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = VAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)

    map_location = "cuda" if cfg.policy.cuda and torch.cuda.is_available() else "cpu"

    policy.eval_mode.load_state_dict(
        torch.load(
            CKPT_PATH,
            map_location=map_location,
        )
    )

    print("Loaded checkpoint successfully.")
    print("============================================")

    env = gym.make(cfg.env.env_id)

    results = []

    for ep in range(1, N_EVAL_EPISODES + 1):
        episode_seed = seed + ep * 9973

        try:
            env.seed(episode_seed)
        except Exception:
            pass

        obs = env.reset()

        try:
            policy.eval_mode.reset([0])
        except Exception:
            try:
                policy.eval_mode.reset()
            except Exception:
                pass

        done = False
        total_reward = 0.0
        step_count = 0

        min_ped_dist = float("inf")
        speed_sum = 0.0
        abs_speed_change_sum = 0.0
        previous_speed = None
        previous_action_id = None
        action_switch_count = 0

        last_info = {}

        video_path = os.path.join(
            replay_dir,
            f"episode_{ep:03d}_seed_{episode_seed}.mp4"
        )

        writer = None
        if SAVE_VIDEO and imageio is not None:
            try:
                writer = imageio.get_writer(video_path, fps=VIDEO_FPS)
            except Exception as err:
                print("Could not create video writer:", err)
                writer = None

        while not done:
            obs_np = np.asarray(obs, dtype=np.float32)

            with torch.no_grad():
                try:
                    policy_output = policy.eval_mode.forward({0: obs_np})
                except Exception:
                    obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32)
                    policy_output = policy.eval_mode.forward({0: obs_tensor})

            action_id, action_params = extract_hybrid_action(policy_output)
            obs, reward, done, info = env.step((action_id, action_params))

            total_reward += float(reward)
            step_count += 1
            last_info = info if isinstance(info, dict) else {}

            current_min_ped = nearest_pedestrian_distance(env)
            min_ped_dist = min(min_ped_dist, current_min_ped)

            current_speed = abs(float(env.agent.speed))
            speed_sum += current_speed

            if previous_speed is not None:
                abs_speed_change_sum += abs(current_speed - previous_speed)

            previous_speed = current_speed

            if previous_action_id is not None and action_id != previous_action_id:
                action_switch_count += 1

            previous_action_id = action_id

            if writer is not None:
                if step_count % VIDEO_FRAME_SKIP == 0 or done:
                    try:
                        frame = env.render(mode="rgb_array")
                        if frame is not None:
                            writer.append_data(frame)
                    except Exception as err:
                        print("Video render failed:", err)
                        try:
                            writer.close()
                        except Exception:
                            pass
                        writer = None

        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass

        done_reason = infer_done_reason(env, last_info)

        episode_time = step_count * float(env.delta_t)
        avg_speed = speed_sum / max(step_count, 1)
        avg_abs_speed_change = abs_speed_change_sum / max(step_count - 1, 1)
        action_switch_rate = action_switch_count / max(step_count - 1, 1)

        social_norm_count = get_info_value(last_info, "social_norm_count", 0)
        social_norm_rate = get_info_value(last_info, "social_norm_rate", 0.0)

        result = {
            "episode": ep,
            "done_reason": done_reason,
            "reward": total_reward,
            "steps": step_count,
            "time": episode_time,
            "min_pedestrian_distance": min_ped_dist,
            "avg_speed": avg_speed,
            "avg_abs_speed_change": avg_abs_speed_change,
            "action_switch_count": action_switch_count,
            "action_switch_rate": action_switch_rate,
            "social_norm_count": social_norm_count,
            "social_norm_rate": social_norm_rate,
            "video_path": video_path if SAVE_VIDEO else "",
        }

        results.append(result)

        goal_so_far = sum(1 for r in results if r["done_reason"] == "goal")
        collision_so_far = sum(1 for r in results if r["done_reason"] == "collision")
        boundary_so_far = sum(1 for r in results if r["done_reason"] == "boundary")
        timeout_so_far = sum(1 for r in results if r["done_reason"] == "timeout")
        unknown_so_far = sum(1 for r in results if r["done_reason"] == "unknown")

        print("--------------------------------------------")
        print(f"Episode {ep}/{N_EVAL_EPISODES} finished")
        print(f"Done reason: {done_reason}")
        print(f"Reward: {total_reward:.4f}")
        print(f"Steps: {step_count}")
        print(f"Time: {episode_time:.4f}")
        print(f"Minimum pedestrian distance: {min_ped_dist:.4f}")
        print(f"Average speed: {avg_speed:.4f}")
        print(f"Average abs speed change: {avg_abs_speed_change:.6f}")
        print(f"Action switch count: {action_switch_count}")
        print(f"Action switch rate: {action_switch_rate:.4f}")
        print(f"Social norm count: {social_norm_count}")
        print(f"Social norm rate: {float(social_norm_rate):.4f}")
        if SAVE_VIDEO:
            print(f"Video: {video_path}")
        print("")
        print("Current 12-episode progress:")
        print(f"Goal: {goal_so_far}/{ep}")
        print(f"Collision: {collision_so_far}/{ep}")
        print(f"Boundary: {boundary_so_far}/{ep}")
        print(f"Timeout: {timeout_so_far}/{ep}")
        print(f"Unknown: {unknown_so_far}/{ep}")

    env.close()

    total = len(results)

    goal_count = sum(1 for r in results if r["done_reason"] == "goal")
    collision_count = sum(1 for r in results if r["done_reason"] == "collision")
    boundary_count = sum(1 for r in results if r["done_reason"] == "boundary")
    timeout_count = sum(1 for r in results if r["done_reason"] == "timeout")
    unknown_count = sum(1 for r in results if r["done_reason"] == "unknown")

    goal_results = [r for r in results if r["done_reason"] == "goal"]

    avg_reward = mean([r["reward"] for r in results])
    avg_steps_all = mean([r["steps"] for r in results])
    avg_time_all = mean([r["time"] for r in results])
    avg_steps_goal = mean([r["steps"] for r in goal_results])
    avg_time_goal = mean([r["time"] for r in goal_results])
    avg_min_ped = mean([r["min_pedestrian_distance"] for r in results])
    avg_speed = mean([r["avg_speed"] for r in results])
    avg_abs_speed_change = mean([r["avg_abs_speed_change"] for r in results])
    avg_action_switch_rate = mean([r["action_switch_rate"] for r in results])
    avg_social_norm_rate = mean([r["social_norm_rate"] for r in results])

    print("\n============================================")
    print("FINAL 12-EPISODE SUMMARY")
    print("============================================")
    print(f"Episodes tested: {total}")
    print(f"Goal-reaching success rate: {goal_count}/{total} = {100.0 * goal_count / total:.2f}%")
    print(f"Collision failure rate: {collision_count}/{total} = {100.0 * collision_count / total:.2f}%")
    print(f"Boundary violation rate: {boundary_count}/{total} = {100.0 * boundary_count / total:.2f}%")
    print(f"Timeout / stuck failure rate: {timeout_count}/{total} = {100.0 * timeout_count / total:.2f}%")
    print(f"Unknown rate: {unknown_count}/{total} = {100.0 * unknown_count / total:.2f}%")
    print("--------------------------------------------")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Average steps, all episodes: {avg_steps_all:.2f}")
    print(f"Average time, all episodes: {avg_time_all:.4f}")
    print(f"Average steps to reach goal only: {avg_steps_goal:.2f}")
    print(f"Average time to reach goal only: {avg_time_goal:.4f}")
    print(f"Average minimum distance to pedestrians: {avg_min_ped:.4f}")
    print(f"Average speed: {avg_speed:.4f}")
    print(f"Average smoothness, abs speed change: {avg_abs_speed_change:.6f}")
    print(f"Average smoothness, action switch rate: {avg_action_switch_rate:.4f}")
    print(f"Average social norm compliance rate: {avg_social_norm_rate:.4f}")
    print("--------------------------------------------")
    print(f"Videos saved to: {replay_dir}")
    print("============================================")


if __name__ == "__main__":
    main(
        AutonomousRobotNavigation2_hppo_config,
        AutonomousRobotNavigation2_hppo_create_config,
        seed=0,
    )
