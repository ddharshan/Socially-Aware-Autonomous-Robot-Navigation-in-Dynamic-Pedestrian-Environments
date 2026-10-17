import os
import time

import gym
import torch
import numpy as np

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

# Important: this registers DRLRobot-v0
import AutonomousRobotNavigation2  # noqa: F401


try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


# ============================================================
# TEST SETTINGS
# ============================================================

EPISODES = 30


# ============================================================
# PATH-71 CHECKPOINT
# ============================================================

CKPT_DIR = (
    r"D:\MSc Robot Navigation - Github\RLRobotTraining2"
    r"\AutonomousRobotNavigation2_hppo_seed0-the best path-71-good"
    r"\ckpt"
)

CKPT_PATH = os.path.join(
    CKPT_DIR,
    "ckpt_best.pth.tar"
)


# ============================================================
# OUTPUT FOLDERS
#
# BOTH folders will be created INSIDE the ckpt folder:
#
# ckpt
# ├── MP4_30_Episodes
# └── GIF_30_Episodes
# ============================================================

MP4_DIR = os.path.join(
    CKPT_DIR,
    "MP4_30_Episodes"
)

GIF_DIR = os.path.join(
    CKPT_DIR,
    "GIF_30_Episodes"
)


# ============================================================
# VIDEO SETTINGS
# ============================================================

VIDEO_FPS = 30

# Save every simulation frame to MP4
VIDEO_FRAME_SKIP = 1


# ============================================================
# GIF SETTINGS
# ============================================================

# GIF is intentionally lighter than MP4
GIF_FPS = 15

# Save every second frame to GIF
GIF_FRAME_SKIP = 2


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_base_env(env):
    """
    Remove Gym wrappers and return the underlying environment.
    """
    while hasattr(env, "env"):
        env = env.env

    return env


def to_numpy(x):
    """
    Convert Torch tensor or similar object to NumPy.
    """
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def scale_from_minus1_plus1(
    x,
    min_val,
    max_val
):
    """
    Scale a value from [-1, +1] to [min_val, max_val].
    """

    x = float(x)

    x = np.clip(
        x,
        -1.0,
        1.0
    )

    return (
        min_val
        + (x + 1.0)
        * 0.5
        * (max_val - min_val)
    )


def extract_hybrid_action(
    policy_output,
    act_scale=True
):
    """
    Convert DI-engine hybrid PPO output to the Gym action:

        (
            action_id,
            action_parameters
        )

    action_id:
        0 = accelerate
        1 = turn
        2 = brake

    action_parameters:
        [
            acceleration_parameter,
            turn_parameter
        ]
    """

    # --------------------------------------------------------
    # Extract environment 0 output
    # --------------------------------------------------------

    if isinstance(
        policy_output,
        dict
    ):

        if 0 in policy_output:
            out = policy_output[0]

        else:
            out = list(
                policy_output.values()
            )[0]

    else:
        out = policy_output


    # --------------------------------------------------------
    # Extract action
    # --------------------------------------------------------

    if "action" not in out:

        print(
            "Policy output:",
            out
        )

        raise KeyError(
            "Cannot find 'action' in policy output."
        )

    action = out["action"]


    # --------------------------------------------------------
    # Dictionary-format action
    # --------------------------------------------------------

    if isinstance(
        action,
        dict
    ):

        action_type = action.get(
            "action_type",
            None
        )

        action_args = action.get(
            "action_args",
            None
        )

        if action_type is None:
            action_type = action.get(
                "type",
                None
            )

        if action_args is None:
            action_args = action.get(
                "args",
                None
            )

        if (
            action_type is None
            or action_args is None
        ):

            print(
                "Action:",
                action
            )

            raise KeyError(
                "Cannot find action_type/action_args."
            )

        action_id = int(
            to_numpy(
                action_type
            ).reshape(-1)[0]
        )

        action_params = (
            to_numpy(
                action_args
            )
            .astype(np.float32)
            .reshape(-1)
        )


    # --------------------------------------------------------
    # Tuple/list action
    # --------------------------------------------------------

    elif isinstance(
        action,
        (tuple, list)
    ):

        action_id = int(
            to_numpy(
                action[0]
            ).reshape(-1)[0]
        )

        action_params = (
            to_numpy(
                action[1]
            )
            .astype(np.float32)
            .reshape(-1)
        )

    else:

        raise TypeError(
            f"Unsupported action format: {type(action)}"
        )


    # --------------------------------------------------------
    # Action-ID protection
    # --------------------------------------------------------

    action_id = int(
        np.clip(
            action_id,
            0,
            2
        )
    )


    # --------------------------------------------------------
    # Ensure two continuous parameters
    # --------------------------------------------------------

    if action_params.size < 2:

        padded = np.zeros(
            2,
            dtype=np.float32
        )

        padded[
            :action_params.size
        ] = action_params

        action_params = padded

    action_params = (
        action_params[:2]
        .astype(np.float32)
    )


    # --------------------------------------------------------
    # Scale actions exactly as required by wrapper
    # --------------------------------------------------------

    if act_scale:

        # Acceleration:
        # [-1, 1] -> [0, 1]

        action_params[0] = (
            scale_from_minus1_plus1(
                action_params[0],
                0.0,
                1.0
            )
        )

        # Turning:
        # [-1, 1] -> [-1, 1]

        action_params[1] = (
            scale_from_minus1_plus1(
                action_params[1],
                -1.0,
                1.0
            )
        )

    else:

        action_params[0] = np.clip(
            action_params[0],
            0.0,
            1.0
        )

        action_params[1] = np.clip(
            action_params[1],
            -1.0,
            1.0
        )

    return (
        action_id,
        action_params
    )


# ============================================================
# PEDESTRIAN DISTANCE
# ============================================================

def nearest_pedestrian_distance(
    base_env
):
    """
    Return distance to nearest of six pedestrians.
    """

    return min(
        base_env.collision1,
        base_env.collision2,
        base_env.collision3,
        base_env.collision4,
        base_env.collision5,
        base_env.collision6,
    )


# ============================================================
# TERMINATION REASON
# ============================================================

def infer_done_reason(
    base_env,
    info
):
    """
    Prefer done_reason supplied by environment.

    Otherwise manually determine:
        goal
        boundary
        timeout
        collision
        unknown
    """

    if (
        isinstance(info, dict)
        and "done_reason" in info
    ):

        return info[
            "done_reason"
        ]


    # Goal
    if (
        base_env.distance
        <= base_env.target_radius
    ):

        return "goal"


    # Boundary
    if (
        abs(base_env.agent.x)
        > base_env.field_size
        or
        abs(base_env.agent.y)
        > base_env.field_size
    ):

        return "boundary"


    # Timeout
    if (
        base_env.current_step
        > base_env.max_step
    ):

        return "timeout"


    # Collision
    if (
        nearest_pedestrian_distance(
            base_env
        )
        <= 0.21
    ):

        return "collision"


    return "unknown"


# ============================================================
# SOCIAL NORM MONITORING
# ============================================================

def get_theta_norm(theta):

    if theta > np.pi:

        return (
            -2.0 * np.pi
            + theta
        )

    return theta


def social_norm_step(
    base_env
):
    """
    Backup social-norm monitor.

    This does NOT modify the reward.

    It only measures social-norm opportunities/compliance
    during evaluation.
    """

    if (
        base_env.distance
        <= 0.4
    ):

        return (
            False,
            False
        )


    robot_theta = get_theta_norm(
        base_env.agent.theta
    )

    robot_speed = abs(
        base_env.agent.speed
    )


    pedestrians = [

        (
            base_env.collision1,
            get_theta_norm(
                base_env.pedestrian1.p1theta
            ),
            abs(
                base_env.pedestrian1.p1speed
            )
        ),

        (
            base_env.collision2,
            get_theta_norm(
                base_env.pedestrian2.p2theta
            ),
            abs(
                base_env.pedestrian2.p2speed
            )
        ),

        (
            base_env.collision3,
            get_theta_norm(
                base_env.pedestrian3.p3theta
            ),
            abs(
                base_env.pedestrian3.p3speed
            )
        ),

        (
            base_env.collision4,
            get_theta_norm(
                base_env.pedestrian4.p4theta
            ),
            abs(
                base_env.pedestrian4.p4speed
            )
        ),

        (
            base_env.collision5,
            get_theta_norm(
                base_env.pedestrian5.p5theta
            ),
            abs(
                base_env.pedestrian5.p5speed
            )
        ),

        (
            base_env.collision6,
            get_theta_norm(
                base_env.pedestrian6.p6theta
            ),
            abs(
                base_env.pedestrian6.p6speed
            )
        ),
    ]


    for (
        collision_dist,
        ped_theta,
        ped_speed
    ) in pedestrians:

        # Final social-norm distance zone:
        # 0.30 <= distance < 0.35

        if (
            0.30
            <= collision_dist
            < 0.35
        ):

            passing = (
                (0.75 * np.pi)
                < abs(
                    ped_theta
                    - robot_theta
                )
                < np.pi
            )

            crossing = (

                (np.pi / 4)
                < (
                    ped_theta
                    - robot_theta
                )
                < (0.75 * np.pi)

                and

                (
                    ped_speed
                    - robot_speed
                    > 0
                )
            )

            overtaking = (

                0
                < (
                    robot_theta
                    - ped_theta
                )
                < (np.pi / 4)

                and

                (
                    robot_speed
                    > ped_speed
                )
            )

            compliant = (
                passing
                or crossing
                or overtaking
            )

            return (
                True,
                compliant
            )

    return (
        False,
        False
    )


# ============================================================
# MEAN
# ============================================================

def mean(values):

    values = [
        v
        for v in values
        if v is not None
    ]

    if len(values) == 0:
        return 0.0

    return float(
        np.mean(values)
    )


# ============================================================
# RESET POLICY STATE
# ============================================================

def reset_policy(policy):

    try:

        policy.eval_mode.reset(
            [0]
        )

    except Exception:

        try:

            policy.eval_mode.reset()

        except Exception:

            pass


# ============================================================
# CHECKPOINT LOADER
# ============================================================

def load_checkpoint(
    policy,
    ckpt_path,
    map_location
):

    print("")
    print(
        "Loading checkpoint..."
    )

    ckpt_data = torch.load(
        ckpt_path,
        map_location=map_location
    )


    # --------------------------------------------------------
    # First attempt
    # --------------------------------------------------------

    try:

        policy.eval_mode.load_state_dict(
            ckpt_data
        )

        return

    except Exception:
        pass


    # --------------------------------------------------------
    # Alternative checkpoint formats
    # --------------------------------------------------------

    if isinstance(
        ckpt_data,
        dict
    ):

        for key in [

            "model",
            "state_dict",
            "policy",
            "model_state_dict"

        ]:

            if key in ckpt_data:

                try:

                    policy.eval_mode.load_state_dict(
                        ckpt_data[key]
                    )

                    return

                except Exception:

                    pass


    raise RuntimeError(
        "Could not load checkpoint into policy.eval_mode."
    )


# ============================================================
# MP4 WRITER
# ============================================================

def create_mp4_writer(
    video_path
):

    if imageio is None:

        raise RuntimeError(
            "imageio is not installed."
        )

    try:

        return imageio.get_writer(
            video_path,
            fps=VIDEO_FPS,
            macro_block_size=1
        )

    except TypeError:

        return imageio.get_writer(
            video_path,
            fps=VIDEO_FPS
        )


# ============================================================
# GIF WRITER
# ============================================================

def create_gif_writer(
    gif_path
):

    if imageio is None:

        raise RuntimeError(
            "imageio is not installed."
        )

    return imageio.get_writer(
        gif_path,
        mode="I",
        duration=1.0 / GIF_FPS,
        loop=0
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(
    results
):

    total = len(results)

    goal_count = sum(
        1
        for r in results
        if r["done_reason"] == "goal"
    )

    collision_count = sum(
        1
        for r in results
        if r["done_reason"] == "collision"
    )

    boundary_count = sum(
        1
        for r in results
        if r["done_reason"] == "boundary"
    )

    timeout_count = sum(
        1
        for r in results
        if r["done_reason"] == "timeout"
    )

    unknown_count = sum(
        1
        for r in results
        if r["done_reason"] == "unknown"
    )


    goal_results = [

        r

        for r in results

        if r["done_reason"] == "goal"
    ]


    total_social_norm_opportunities = sum(

        r[
            "social_norm_opportunity_count"
        ]

        for r in results
    )


    total_social_norm_compliant = sum(

        r[
            "social_norm_compliant_count"
        ]

        for r in results
    )


    if (
        total_social_norm_opportunities
        > 0
    ):

        overall_social_norm_compliance_rate = (

            total_social_norm_compliant

            /

            total_social_norm_opportunities
        )

    else:

        overall_social_norm_compliance_rate = 0.0


    avg_reward = mean(
        [
            r["reward"]
            for r in results
        ]
    )

    avg_steps_all = mean(
        [
            r["steps"]
            for r in results
        ]
    )

    avg_time_all = mean(
        [
            r["time"]
            for r in results
        ]
    )

    avg_steps_goal = mean(
        [
            r["steps"]
            for r in goal_results
        ]
    )

    avg_time_goal = mean(
        [
            r["time"]
            for r in goal_results
        ]
    )

    avg_min_ped = mean(
        [
            r["min_pedestrian_distance"]
            for r in results
        ]
    )

    global_min_ped = min(
        [
            r["min_pedestrian_distance"]
            for r in results
        ]
    )

    avg_speed = mean(
        [
            r["avg_speed"]
            for r in results
        ]
    )

    avg_abs_speed_change = mean(
        [
            r["avg_abs_speed_change"]
            for r in results
        ]
    )

    avg_action_switch_rate = mean(
        [
            r["action_switch_rate"]
            for r in results
        ]
    )


    print("")
    print(
        "============================================================"
    )

    print(
        "FINAL 30-EPISODE PATH-71 TEST SUMMARY"
    )

    print(
        "============================================================"
    )

    print(
        f"Episodes tested: {total}"
    )

    print(
        f"Goal-reaching success rate: "
        f"{goal_count}/{total} = "
        f"{100.0 * goal_count / total:.2f}%"
    )

    print(
        f"Collision failure rate: "
        f"{collision_count}/{total} = "
        f"{100.0 * collision_count / total:.2f}%"
    )

    print(
        f"Boundary violation rate: "
        f"{boundary_count}/{total} = "
        f"{100.0 * boundary_count / total:.2f}%"
    )

    print(
        f"Timeout / stuck failure rate: "
        f"{timeout_count}/{total} = "
        f"{100.0 * timeout_count / total:.2f}%"
    )

    print(
        f"Unknown rate: "
        f"{unknown_count}/{total} = "
        f"{100.0 * unknown_count / total:.2f}%"
    )

    print(
        "------------------------------------------------------------"
    )

    print(
        f"Average reward: "
        f"{avg_reward:.4f}"
    )

    print(
        f"Average steps, all episodes: "
        f"{avg_steps_all:.2f}"
    )

    print(
        f"Average time, all episodes: "
        f"{avg_time_all:.4f}"
    )

    print(
        f"Average steps to reach goal only: "
        f"{avg_steps_goal:.2f}"
    )

    print(
        f"Average time to reach goal only: "
        f"{avg_time_goal:.4f}"
    )

    print(
        f"Average minimum distance to pedestrians: "
        f"{avg_min_ped:.4f}"
    )

    print(
        f"Global minimum distance to pedestrians: "
        f"{global_min_ped:.4f}"
    )

    print(
        "------------------------------------------------------------"
    )

    print(
        f"Total social norm opportunities: "
        f"{total_social_norm_opportunities}"
    )

    print(
        f"Total social norm compliant steps: "
        f"{total_social_norm_compliant}"
    )

    print(
        f"Social norm compliance rate: "
        f"{100.0 * overall_social_norm_compliance_rate:.2f}%"
    )

    print(
        "------------------------------------------------------------"
    )

    print(
        f"Average speed: "
        f"{avg_speed:.4f}"
    )

    print(
        f"Average smoothness, abs speed change: "
        f"{avg_abs_speed_change:.6f}"
    )

    print(
        f"Average smoothness, action switch rate: "
        f"{avg_action_switch_rate:.4f}"
    )

    print(
        "============================================================"
    )


# ============================================================
# MAIN TEST
# ============================================================

def main(
    main_cfg,
    create_cfg
):

    # --------------------------------------------------------
    # New random seed each execution
    # --------------------------------------------------------

    seed = int(
        time.time() * 1000
    ) % 1000000


    # --------------------------------------------------------
    # Verify checkpoint
    # --------------------------------------------------------

    if not os.path.exists(
        CKPT_PATH
    ):

        raise FileNotFoundError(

            "\nCheckpoint was not found:\n"
            f"{CKPT_PATH}\n"
        )


    # --------------------------------------------------------
    # Create output folders
    # --------------------------------------------------------

    os.makedirs(
        MP4_DIR,
        exist_ok=True
    )

    os.makedirs(
        GIF_DIR,
        exist_ok=True
    )


    if imageio is None:

        raise RuntimeError(

            "imageio is not installed.\n"
            "Run:\n"
            "python -m pip install imageio imageio-ffmpeg pillow"
        )


    # --------------------------------------------------------
    # Configure evaluator
    # --------------------------------------------------------

    main_cfg.env.evaluator_env_num = 1

    main_cfg.env.n_evaluator_episode = (
        EPISODES
    )


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


    # --------------------------------------------------------
    # Hybrid PPO command mode
    # --------------------------------------------------------

    if not create_cfg.policy.type.endswith(
        "_command"
    ):

        create_cfg.policy.type = (
            create_cfg.policy.type
            + "_command"
        )


    cfg = compile_config(

        cfg,

        seed=seed,

        env=None,

        auto=True,

        create_cfg=create_cfg,

        save_cfg=True,
    )


    # --------------------------------------------------------
    # Test information
    # --------------------------------------------------------

    print("")
    print(
        "============================================================"
    )

    print(
        "PATH-71 : 30-EPISODE TRAINED POLICY TEST"
    )

    print(
        "============================================================"
    )

    print("")
    print(
        "Checkpoint:"
    )

    print(
        CKPT_PATH
    )

    print("")
    print(
        f"Random test seed: {seed}"
    )

    print(
        f"Number of episodes: {EPISODES}"
    )

    print("")
    print(
        "MP4 output folder:"
    )

    print(
        MP4_DIR
    )

    print("")
    print(
        "GIF output folder:"
    )

    print(
        GIF_DIR
    )

    print("")
    print(
        "Each MP4 and GIF pair represents the SAME episode."
    )

    print(
        "Every episode uses a different random environment seed."
    )

    print(
        "============================================================"
    )


    # --------------------------------------------------------
    # Seed DI-engine
    # --------------------------------------------------------

    set_pkg_seed(
        seed,
        use_cuda=cfg.policy.cuda
    )


    # --------------------------------------------------------
    # Build PPO model
    # --------------------------------------------------------

    model = VAC(
        **cfg.policy.model
    )

    policy = PPOPolicy(
        cfg.policy,
        model=model
    )


    # --------------------------------------------------------
    # CPU/GPU checkpoint loading
    # --------------------------------------------------------

    map_location = (

        "cuda"

        if (
            cfg.policy.cuda
            and torch.cuda.is_available()
        )

        else

        "cpu"
    )


    # --------------------------------------------------------
    # Load path-71 checkpoint
    # --------------------------------------------------------

    load_checkpoint(

        policy,

        CKPT_PATH,

        map_location
    )


    print("")
    print(
        "Path-71 checkpoint loaded successfully."
    )

    print(
        "============================================================"
    )


    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------

    env = gym.make(
        cfg.env.env_id
    )


    act_scale = bool(
        cfg.env.act_scale
    )


    results = []


    # ========================================================
    # 30 EPISODES
    # ========================================================

    for ep in range(
        1,
        EPISODES + 1
    ):

        # ----------------------------------------------------
        # Different random seed for every episode
        # ----------------------------------------------------

        episode_seed = (
            seed
            + ep * 9973
        )


        try:

            env.seed(
                episode_seed
            )

        except Exception:

            pass


        # ----------------------------------------------------
        # Reset environment
        # ----------------------------------------------------

        reset_output = env.reset()


        if isinstance(
            reset_output,
            tuple
        ):

            obs = reset_output[0]

        else:

            obs = reset_output


        reset_policy(
            policy
        )


        done = False

        total_reward = 0.0

        step_count = 0


        min_ped_dist = float(
            "inf"
        )


        speed_sum = 0.0

        abs_speed_change_sum = 0.0

        previous_speed = None


        previous_action_id = None

        action_switch_count = 0


        social_norm_opportunity_count = 0

        social_norm_compliant_count = 0


        last_info = {}


        # ----------------------------------------------------
        # File names
        # ----------------------------------------------------

        file_base = (
            f"episode_{ep:03d}"
            f"_seed_{episode_seed}"
        )


        video_path = os.path.join(

            MP4_DIR,

            file_base + ".mp4"
        )


        gif_path = os.path.join(

            GIF_DIR,

            file_base + ".gif"
        )


        # ----------------------------------------------------
        # Create MP4 + GIF writers
        # ----------------------------------------------------

        mp4_writer = create_mp4_writer(
            video_path
        )

        gif_writer = create_gif_writer(
            gif_path
        )


        print("")
        print(
            "------------------------------------------------------------"
        )

        print(
            f"Starting episode {ep}/{EPISODES}"
        )

        print(
            f"Episode seed: {episode_seed}"
        )

        print(
            "------------------------------------------------------------"
        )


        # ====================================================
        # EPISODE LOOP
        # ====================================================

        while not done:

            obs_np = np.asarray(
                obs,
                dtype=np.float32
            )


            # ------------------------------------------------
            # PPO inference
            # ------------------------------------------------

            with torch.no_grad():

                try:

                    policy_output = (
                        policy.eval_mode.forward(
                            {
                                0: obs_np
                            }
                        )
                    )

                except Exception:

                    obs_tensor = torch.as_tensor(

                        obs_np,

                        dtype=torch.float32
                    )

                    policy_output = (
                        policy.eval_mode.forward(
                            {
                                0: obs_tensor
                            }
                        )
                    )


            # ------------------------------------------------
            # Extract trained PPO action
            # ------------------------------------------------

            (
                action_id,
                action_params

            ) = extract_hybrid_action(

                policy_output,

                act_scale=act_scale
            )


            # ------------------------------------------------
            # Environment step
            # ------------------------------------------------

            step_output = env.step(

                (
                    action_id,
                    action_params
                )
            )


            # Gym API compatibility
            if len(
                step_output
            ) == 5:

                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    info

                ) = step_output

                done = bool(
                    terminated
                    or truncated
                )

            else:

                (
                    obs,
                    reward,
                    done,
                    info

                ) = step_output


            # ------------------------------------------------
            # Reward
            # ------------------------------------------------

            reward_float = float(

                np.asarray(
                    reward
                ).reshape(-1)[0]
            )


            total_reward += (
                reward_float
            )


            step_count += 1


            last_info = (

                info

                if isinstance(
                    info,
                    dict
                )

                else
                {}
            )


            # ------------------------------------------------
            # Underlying environment
            # ------------------------------------------------

            base_env = get_base_env(
                env
            )


            # ------------------------------------------------
            # Minimum pedestrian distance
            # ------------------------------------------------

            current_min_ped = (
                nearest_pedestrian_distance(
                    base_env
                )
            )


            min_ped_dist = min(

                min_ped_dist,

                current_min_ped
            )


            # ------------------------------------------------
            # Speed metrics
            # ------------------------------------------------

            current_speed = abs(

                float(
                    base_env.agent.speed
                )
            )


            speed_sum += (
                current_speed
            )


            if (
                previous_speed
                is not None
            ):

                abs_speed_change_sum += abs(

                    current_speed
                    - previous_speed
                )


            previous_speed = (
                current_speed
            )


            # ------------------------------------------------
            # Action switches
            # ------------------------------------------------

            if (

                previous_action_id
                is not None

                and

                action_id
                != previous_action_id
            ):

                action_switch_count += 1


            previous_action_id = (
                action_id
            )


            # ------------------------------------------------
            # Social norm metrics
            # ------------------------------------------------

            (
                norm_opportunity,
                norm_compliant

            ) = social_norm_step(
                base_env
            )


            if norm_opportunity:

                social_norm_opportunity_count += 1


            if norm_compliant:

                social_norm_compliant_count += 1


            # =================================================
            # RENDER ONLY ONCE
            #
            # The same RGB frame is then used for both
            # MP4 and GIF.
            # =================================================

            try:

                frame = env.render(
                    mode="rgb_array"
                )


                if frame is not None:

                    frame = np.asarray(
                        frame
                    )


                    # -----------------------------------------
                    # MP4
                    # -----------------------------------------

                    if (

                        step_count
                        % VIDEO_FRAME_SKIP
                        == 0

                        or done
                    ):

                        mp4_writer.append_data(
                            frame
                        )


                    # -----------------------------------------
                    # GIF
                    # -----------------------------------------

                    if (

                        step_count
                        % GIF_FRAME_SKIP
                        == 0

                        or done
                    ):

                        gif_writer.append_data(
                            frame
                        )


            except Exception as err:

                print("")
                print(
                    "Rendering failed:"
                )

                print(
                    err
                )

                # Close writers before stopping
                try:

                    mp4_writer.close()

                except Exception:

                    pass


                try:

                    gif_writer.close()

                except Exception:

                    pass


                env.close()

                raise


        # ====================================================
        # CLOSE VIDEO WRITERS AFTER EACH EPISODE
        # ====================================================

        try:

            mp4_writer.close()

        except Exception as err:

            print(
                "MP4 closing warning:",
                err
            )


        try:

            gif_writer.close()

        except Exception as err:

            print(
                "GIF closing warning:",
                err
            )


        # ====================================================
        # EPISODE RESULT
        # ====================================================

        base_env = get_base_env(
            env
        )


        done_reason = infer_done_reason(

            base_env,

            last_info
        )


        episode_time = (

            step_count

            * float(
                base_env.delta_t
            )
        )


        avg_speed = (

            speed_sum

            /

            max(
                step_count,
                1
            )
        )


        avg_abs_speed_change = (

            abs_speed_change_sum

            /

            max(
                step_count - 1,
                1
            )
        )


        action_switch_rate = (

            action_switch_count

            /

            max(
                step_count - 1,
                1
            )
        )


        if (

            social_norm_opportunity_count
            > 0
        ):

            social_norm_compliance_rate = (

                social_norm_compliant_count

                /

                social_norm_opportunity_count
            )

        else:

            social_norm_compliance_rate = (
                0.0
            )


        result = {

            "episode":
                ep,

            "done_reason":
                done_reason,

            "reward":
                total_reward,

            "steps":
                step_count,

            "time":
                episode_time,

            "min_pedestrian_distance":
                min_ped_dist,

            "avg_speed":
                avg_speed,

            "avg_abs_speed_change":
                avg_abs_speed_change,

            "action_switch_count":
                action_switch_count,

            "action_switch_rate":
                action_switch_rate,

            "social_norm_opportunity_count":
                social_norm_opportunity_count,

            "social_norm_compliant_count":
                social_norm_compliant_count,

            "social_norm_compliance_rate":
                social_norm_compliance_rate,

            "video_path":
                video_path,

            "gif_path":
                gif_path,
        }


        results.append(
            result
        )


        # ====================================================
        # PRINT EPISODE RESULT
        # ====================================================

        goal_so_far = sum(

            1

            for r in results

            if r["done_reason"]
            == "goal"
        )


        collision_so_far = sum(

            1

            for r in results

            if r["done_reason"]
            == "collision"
        )


        boundary_so_far = sum(

            1

            for r in results

            if r["done_reason"]
            == "boundary"
        )


        timeout_so_far = sum(

            1

            for r in results

            if r["done_reason"]
            == "timeout"
        )


        unknown_so_far = sum(

            1

            for r in results

            if r["done_reason"]
            == "unknown"
        )


        print("")
        print(
            "============================================================"
        )

        print(
            f"Episode {ep}/{EPISODES} finished"
        )

        print(
            "============================================================"
        )

        print(
            f"Done reason: {done_reason}"
        )

        print(
            f"Reward: {total_reward:.4f}"
        )

        print(
            f"Steps: {step_count}"
        )

        print(
            f"Time: {episode_time:.4f}"
        )

        print(
            f"Minimum pedestrian distance: "
            f"{min_ped_dist:.4f}"
        )

        print(
            f"Average speed: "
            f"{avg_speed:.4f}"
        )

        print(
            f"Average abs speed change: "
            f"{avg_abs_speed_change:.6f}"
        )

        print(
            f"Action switch rate: "
            f"{action_switch_rate:.4f}"
        )

        print(
            f"Social norm opportunities: "
            f"{social_norm_opportunity_count}"
        )

        print(
            f"Social norm compliant steps: "
            f"{social_norm_compliant_count}"
        )

        print(
            f"Social norm compliance rate: "
            f"{100.0 * social_norm_compliance_rate:.2f}%"
        )

        print("")
        print(
            "MP4 saved:"
        )

        print(
            video_path
        )

        print("")
        print(
            "GIF saved:"
        )

        print(
            gif_path
        )


        print("")
        print(
            "CURRENT 30-EPISODE PROGRESS"
        )

        print(
            "------------------------------------------------------------"
        )

        print(
            f"Goal: "
            f"{goal_so_far}/{ep}"
        )

        print(
            f"Collision: "
            f"{collision_so_far}/{ep}"
        )

        print(
            f"Boundary: "
            f"{boundary_so_far}/{ep}"
        )

        print(
            f"Timeout / stuck: "
            f"{timeout_so_far}/{ep}"
        )

        print(
            f"Unknown: "
            f"{unknown_so_far}/{ep}"
        )

        print(
            "============================================================"
        )


    # ========================================================
    # CLOSE ENVIRONMENT
    # ========================================================

    env.close()


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print_final_summary(
        results
    )


    # ========================================================
    # FINAL OUTPUT LOCATIONS
    # ========================================================

    print("")
    print(
        "============================================================"
    )

    print(
        "ALL 30 EPISODES COMPLETED"
    )

    print(
        "============================================================"
    )

    print("")
    print(
        "MP4 FILES SAVED IN:"
    )

    print(
        MP4_DIR
    )

    print("")
    print(
        "GIF FILES SAVED IN:"
    )

    print(
        GIF_DIR
    )

    print("")
    print(
        "Expected MP4 files: 30"
    )

    print(
        "Expected GIF files: 30"
    )

    print(
        "============================================================"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main(

        AutonomousRobotNavigation2_hppo_config,

        AutonomousRobotNavigation2_hppo_create_config
    )
