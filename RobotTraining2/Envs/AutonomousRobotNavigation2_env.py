from typing import Dict, Optional
import gym
import AutonomousRobotNavigation2  # change-----------------------------------------------------------
import numpy as np
from easydict import EasyDict

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs.common import affine_transform
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('AutonomousRobotNavigation2')  # change----------------------------------------------------------
class AutoRobot2CEnv(BaseEnv):  # change---------------------------------------------------------------------------------------
    default_env_id = ['DRLRobot-v0']

    def __init__(self, cfg: EasyDict) -> None:
        self._cfg = cfg
        self._env_id = cfg.env_id
        assert self._env_id in self.default_env_id
        self._act_scale = cfg.act_scale
        self._init_flag = False
        self._replay_path = None

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = gym.make(self._env_id)

            if self._replay_path is not None:
                if gym.version.VERSION > '0.22.0':
                    # Gym removed classic control rendering to support using pygame instead.
                    # And thus, gym hybrid currently does not support rendering.
                    self._env.metadata["render_modes"] = ["human", "rgb_array"]
                else:
                    self._env = gym.wrappers.RecordVideo(
                        self._env,
                        video_folder=self._replay_path,
                        episode_trigger=lambda episode_id: True,
                        name_prefix='rl-video-{}'.format(id(self))
                    )
                    self._env.metadata["render.modes"] = ["human", "rgb_array"]

            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0],
                high=self._env.reward_range[1],
                shape=(1,),
                dtype=np.float32
            )
            self._init_flag = True

        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)

        # Episode-level metric tracking for DI-engine wrapper.
        self._final_eval_reward = 0.0
        self._episode_step_count = 0
        self._min_pedestrian_distance = float("inf")
        self._speed_sum = 0.0
        self._abs_speed_change_sum = 0.0
        self._previous_speed = None
        self._previous_action_type = None
        self._action_switch_count = 0

        obs = self._env.reset()
        obs = to_ndarray(obs).astype(np.float32)

        assert isinstance(obs, np.ndarray) and obs.shape == (60,)

        return obs

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def _nearest_pedestrian_distance(self) -> float:
        return min(
            self._env.collision1,
            self._env.collision2,
            self._env.collision3,
            self._env.collision4,
            self._env.collision5,
            self._env.collision6,
        )

    def _infer_done_reason(self) -> str:
        if self._env.distance <= self._env.target_radius:
            return "goal"

        if abs(self._env.agent.x) > self._env.field_size or abs(self._env.agent.y) > self._env.field_size:
            return "boundary"

        if self._env.current_step > self._env.max_step:
            return "timeout"

        if self._nearest_pedestrian_distance() <= 0.21:
            return "collision"

        return "unknown"

    def step(self, action: Dict) -> BaseEnvTimestep:
        # Copy action to avoid modifying original DI-engine action object.
        action_type = action['action_type']
        action_args = np.array(action['action_args'], dtype=np.float32)

        if self._act_scale:
            # acceleration_value: scale to [0, 1]
            action_args[0] = affine_transform(
                action_args[0],
                min_val=0,
                max_val=1
            )

            # rotation_value: scale to [-1, 1]
            action_args[1] = affine_transform(
                action_args[1],
                min_val=-1,
                max_val=1
            )

        # Gym environment expects [action_type, action_args]
        gym_action = [action_type, action_args]

        obs, rew, done, info = self._env.step(gym_action)

        # Make sure info is always a dictionary.
        if info is None or not isinstance(info, dict):
            info = {}

        # Convert reward to scalar float.
        rew_float = float(np.asarray(rew).reshape(-1)[0])

        # Accumulate scalar episode return.
        self._final_eval_reward += rew_float

        # --------------------------------------------------------
        # Wrapper-level metric tracking
        # This does not change training behavior.
        # It only records useful information for training/evaluation logs.
        # --------------------------------------------------------
        self._episode_step_count += 1

        nearest_distance = self._nearest_pedestrian_distance()
        self._min_pedestrian_distance = min(
            self._min_pedestrian_distance,
            nearest_distance
        )

        current_speed = abs(float(self._env.agent.speed))
        self._speed_sum += current_speed

        if self._previous_speed is not None:
            self._abs_speed_change_sum += abs(current_speed - self._previous_speed)

        self._previous_speed = current_speed

        action_type_int = int(np.asarray(action_type).reshape(-1)[0])

        if self._previous_action_type is not None and action_type_int != self._previous_action_type:
            self._action_switch_count += 1

        self._previous_action_type = action_type_int

        # If done, attach final episode metrics.
        if done:
            episode_steps = int(getattr(self._env, "current_step", self._episode_step_count))
            episode_time = float(episode_steps * getattr(self._env, "delta_t", 0.0))

            avg_speed = self._speed_sum / max(self._episode_step_count, 1)
            avg_abs_speed_change = self._abs_speed_change_sum / max(self._episode_step_count - 1, 1)
            action_switch_rate = self._action_switch_count / max(self._episode_step_count - 1, 1)

            # DI-engine evaluator needs scalar float, not list or ndarray.
            info['eval_episode_return'] = float(self._final_eval_reward)
            info['final_eval_reward'] = float(self._final_eval_reward)

            # Do not overwrite environment.py values if they already exist.
            info.setdefault('done_reason', self._infer_done_reason())
            info.setdefault('episode_steps', episode_steps)
            info.setdefault('episode_time', episode_time)
            info.setdefault('episode_return', float(self._final_eval_reward))
            info.setdefault('min_pedestrian_distance', float(self._min_pedestrian_distance))
            info.setdefault('avg_speed', float(avg_speed))
            info.setdefault('avg_abs_speed_change', float(avg_abs_speed_change))
            info.setdefault('action_switch_count', int(self._action_switch_count))
            info.setdefault('action_switch_rate', float(action_switch_rate))

            # If environment.py provides these, they will be kept.
            # If not, defaults are added.
            info.setdefault('social_norm_count', 0)
            info.setdefault('social_norm_rate', 0.0)
            info.setdefault('near_pedestrian_count', 0)
            info.setdefault('near_pedestrian_rate', 0.0)

        obs = to_ndarray(obs)

        if isinstance(obs, list):  # corner case
            for i in range(len(obs)):
                if len(obs[i].shape) == 0:
                    obs[i] = np.array([obs[i]])
            obs = np.concatenate(obs)

        assert isinstance(obs, np.ndarray) and obs.shape == (60,)
        obs = obs.astype(np.float32)

        # DI-engine expects reward as ndarray with shape (1,)
        rew = np.array([rew_float], dtype=np.float32)

        assert isinstance(rew, np.ndarray) and rew.shape == (1,)

        info['action_args_mask'] = np.array([[1, 0], [0, 1], [0, 0]])

        return BaseEnvTimestep(obs, rew, done, info)

    def random_action(self) -> Dict:
        # action_type: 0, 1, 2
        # action_args:
        #   - acceleration_value: [0, 1]
        #   - rotation_value: [-1, 1]
        raw_action = self._action_space.sample()
        return {
            'action_type': raw_action[0],
            'action_args': raw_action[1]
        }

    def __repr__(self) -> str:
        return "AutonomousRobotNavigation2 Env"

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space
