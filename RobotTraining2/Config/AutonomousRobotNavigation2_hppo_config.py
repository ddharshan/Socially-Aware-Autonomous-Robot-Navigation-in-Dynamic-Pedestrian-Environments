from easydict import EasyDict


AutonomousRobotNavigation2_hppo_config = dict(
    # Retraining run from path-41.
    # This config runs 12 evaluation episodes during training.
    exp_name='AutonomousRobotNavigation2_hppo_seed0_retrain_from_best77_metrics',

    env=dict(
        # Training data collection environments.
        # Keep this high for faster PPO data collection.
        collector_env_num=16,

        # Evaluation during training.
        # Use 12 during training for faster parallel evaluation.
        # For clean one-by-one video testing, eval.py will override this to 1.
        evaluator_env_num=12,

        act_scale=True,
        env_id='DRLRobot-v0',

        # During each training evaluation, DI-engine evaluates 36 episodes.
        # Later, for publication statistics, use eval.py with 100 or 1000 episodes.
        n_evaluator_episode=36,

        # Keep high so training does not stop early.
        stop_value=9999,
    ),

    policy=dict(
        cuda=False,
        priority=False,
        action_space='hybrid',
        recompute_adv=True,

        model=dict(
            obs_shape=60,
            action_shape=dict(
                action_type_shape=3,
                action_args_shape=2,
            ),
            action_space='hybrid',
            encoder_hidden_size_list=[256, 128, 64, 64],
            sigma_type='fixed',
            fixed_sigma_value=0.3,
            bound_type='tanh',
        ),

        learn=dict(
            # Medium-safe PPO fine-tuning settings.
            # If the policy becomes unstable, reduce learning_rate to 1e-5
            # and entropy_weight to 0.0015.
            epoch_per_collect=1,
            batch_size=320,
            learning_rate=1e-6,
            value_weight=0.5,
            entropy_weight=0.0001,
            clip_ratio=0.005,
            adv_norm=True,
            value_norm=True,

            learner=dict(
                hook=dict(
                    # Load model weights from path-41.
                    load_ckpt_before_run=(
                        r'D:\MSc Robot Navigation\RLRobotTraining2'
                        r'\AutonomousRobotNavigation2_hppo_seed0-the best path-77-bad'
                        r'\ckpt\ckpt_best.pth.tar'
                    )
                )
            ),

            # False = load model weights only, reset optimizer state.
            # Good for fine-tuning after reward/environment metric changes.
            resume_training=False,
        ),

        collect=dict(
            n_sample=int(3200),
            discount_factor=0.99,
            gae_lambda=0.95,
            collector=dict(
                collect_print_freq=1000,
            ),
        ),

        eval=dict(
            evaluator=dict(
                # Training evaluation frequency.
                # Every 100 training iterations, DI-engine evaluates 12 episodes.
                # If training feels slow, increase to 200.
                eval_freq=100,
            ),
        ),
    ),
)

AutonomousRobotNavigation2_hppo_config = EasyDict(AutonomousRobotNavigation2_hppo_config)
main_config = AutonomousRobotNavigation2_hppo_config


AutonomousRobotNavigation2_hppo_create_config = dict(
    env=dict(
        type='AutonomousRobotNavigation2',
        import_names=['RobotTraining2.Envs.AutonomousRobotNavigation2_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)

AutonomousRobotNavigation2_hppo_create_config = EasyDict(AutonomousRobotNavigation2_hppo_create_config)
create_config = AutonomousRobotNavigation2_hppo_create_config


if __name__ == "__main__":
    from ding.entry import serial_pipeline_onpolicy

    serial_pipeline_onpolicy([main_config, create_config], seed=0)
