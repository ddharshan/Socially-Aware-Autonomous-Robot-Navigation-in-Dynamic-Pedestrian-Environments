import pytest
import numpy as np
from easydict import EasyDict

from RobotTraining2.Envs import AutoRobot2CEnv #change-----------------------------------------------------------


@pytest.mark.envtest
class TestAutoRobot2CEnv:  #change-----------------------------------------------------

    def test_naive(self):
        env = AutoRobot2CEnv(EasyDict({'env_id': 'DRLRobot-v0', 'act_scale': False})) #change----------------------------------------------
        env.enable_save_replay('./video')
        env.seed(314, dynamic_seed=False)
        assert env._seed == 314

        obs = env.reset()
        assert obs.shape == (60, )  #OBS-----------------------------------------------------------------------

        for i in range(1000):   #change-----------------------------------------------------
            random_action = env.random_action()
            print('random_action', random_action)

            timestep = env.step(random_action)

            assert isinstance(timestep.obs, np.ndarray)
            assert isinstance(timestep.done, bool)
            assert timestep.obs.shape == (60, )    #OBS-------------------------------------------------------------------
            assert timestep.reward.shape == (1, )
            assert timestep.info['action_args_mask'].shape == (3, 2)

            if timestep.done:
                print('reset env')
                env.reset()

        print(env.observation_space, env.action_space, env.reward_space)
        env.close()
