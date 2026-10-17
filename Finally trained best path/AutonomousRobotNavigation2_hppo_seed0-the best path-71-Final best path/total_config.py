exp_config = {
    'env': {
        'manager': {
            'episode_num': float("inf"),
            'max_retry': 1,
            'step_timeout': None,
            'auto_reset': True,
            'reset_timeout': None,
            'retry_type': 'reset',
            'retry_waiting_time': 0.1,
            'shared_memory': True,
            'copy_on_get': True,
            'context': 'spawn',
            'wait_num': float("inf"),
            'step_wait_timeout': None,
            'connect_timeout': 60,
            'reset_inplace': False,
            'cfg_type': 'SyncSubprocessEnvManagerDict',
            'type': 'subprocess'
        },
        'stop_value': 9999,
        'n_evaluator_episode': 36,
        'type': 'AutonomousRobotNavigation2',
        'import_names': ['RobotTraining2.Envs.AutonomousRobotNavigation2_env'],
        'collector_env_num': 16,
        'evaluator_env_num': 12,
        'act_scale': True,
        'env_id': 'DRLRobot-v0'
    },
    'policy': {
        'model': {
            'obs_shape': 60,
            'action_shape': {
                'action_type_shape': 3,
                'action_args_shape': 2
            },
            'action_space': 'hybrid',
            'encoder_hidden_size_list': [256, 128, 64, 64],
            'sigma_type': 'fixed',
            'fixed_sigma_value': 0.3,
            'bound_type': 'tanh'
        },
        'learn': {
            'learner': {
                'train_iterations': 1000000000,
                'dataloader': {
                    'num_workers': 0
                },
                'log_policy': True,
                'hook': {
                    'load_ckpt_before_run':
                    'D:\\MSc Robot Navigation\\RLRobotTraining2\\AutonomousRobotNavigation2_hppo_seed0-the best path-66-good\\ckpt\\ckpt_best.pth.tar',
                    'log_show_after_iter': 100,
                    'save_ckpt_after_iter': 10000,
                    'save_ckpt_after_run': True
                },
                'cfg_type': 'BaseLearnerDict'
            },
            'resume_training': False,
            'epoch_per_collect': 1,
            'batch_size': 320,
            'learning_rate': 1e-06,
            'lr_scheduler': None,
            'value_weight': 0.5,
            'entropy_weight': 0.0001,
            'clip_ratio': 0.005,
            'adv_norm': True,
            'value_norm': True,
            'ppo_param_init': True,
            'grad_clip_type': 'clip_norm',
            'grad_clip_value': 0.5,
            'ignore_done': False
        },
        'collect': {
            'collector': {
                'deepcopy_obs': False,
                'transform_obs': False,
                'collect_print_freq': 1000,
                'cfg_type': 'SampleSerialCollectorDict',
                'type': 'sample'
            },
            'unroll_len': 1,
            'discount_factor': 0.99,
            'gae_lambda': 0.95,
            'n_sample': 3200
        },
        'eval': {
            'evaluator': {
                'eval_freq': 100,
                'render': {
                    'render_freq': -1,
                    'mode': 'train_iter'
                },
                'figure_path': None,
                'cfg_type': 'InteractionSerialEvaluatorDict',
                'stop_value': 9999,
                'n_episode': 36
            }
        },
        'other': {
            'replay_buffer': {
                'type': 'advanced',
                'replay_buffer_size': 4096,
                'max_use': float("inf"),
                'max_staleness': float("inf"),
                'alpha': 0.6,
                'beta': 0.4,
                'anneal_step': 100000,
                'enable_track_used_data': False,
                'deepcopy': False,
                'thruput_controller': {
                    'push_sample_rate_limit': {
                        'max': float("inf"),
                        'min': 0
                    },
                    'window_seconds': 30,
                    'sample_min_limit_ratio': 1
                },
                'monitor': {
                    'sampled_data_attr': {
                        'average_range': 5,
                        'print_freq': 200
                    },
                    'periodic_thruput': {
                        'seconds': 60
                    }
                },
                'cfg_type': 'AdvancedReplayBufferDict'
            },
            'commander': {
                'cfg_type': 'BaseSerialCommanderDict'
            }
        },
        'on_policy': True,
        'cuda': False,
        'multi_gpu': False,
        'bp_update_sync': True,
        'traj_len_inf': False,
        'type': 'ppo_command',
        'priority': False,
        'priority_IS_weight': False,
        'recompute_adv': True,
        'action_space': 'hybrid',
        'nstep_return': False,
        'multi_agent': False,
        'transition_with_policy_data': True,
        'cfg_type': 'PPOCommandModePolicyDict'
    },
    'exp_name':
    'AutonomousRobotNavigation2_hppo_seed0_retrain_from_best66_metrics',
    'seed': 0
}
