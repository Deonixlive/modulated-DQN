# import ray.rllib as rllib

from ray import tune, air
import ray

from ray.tune.registry import register_env

from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.dqn import DQN 

import gymnasium as gym
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
#show image
import matplotlib.pyplot as plt

#env creator function for rllib
def atari_env(config):
    import gymnasium as gym
    env = gym.make(**config)
    
    #framestack handled by TrajectoryView API/Model
    env = wrap_deepmind(env)
    return env

# env = atari_env({"id": "ALE/Breakout-v5",
#                 "frameskip": 1})
# plt.imshow(env.reset()[0])
# print(env.spec)
# register_env("atari", env_creator=atari_env)

config = DQNConfig()

replay_config = config.replay_buffer_config

replay_config["capacity"] = 150000
print(replay_config)
config = config.training(
                         gamma=0.99,
                         lr=0.0000625,
                         train_batch_size=32,
                         model={"conv_filters": [[32, [8, 8], 4],
                                                [64, [4, 4], 2],
                                                [64, [3, 3], 1],
                                                [512, [11, 11], 1]]},
                         dueling=True,
                         double_q=True,
                         target_network_update_freq=8000,
                         hiddens=[512],
                         n_step=1,
                         replay_buffer_config=replay_config,
                         # td_error_loss_fn="huber",
                         num_steps_sampled_before_learning_starts=20000,
                         adam_epsilon=0.00015,
                         #DISABLE RAINBOW
                         noisy=False,
                         num_atoms=1,
                        )

config = config.environment(env="atari",
                            env_config={"id": "ALE/Breakout-v5",
                                        "frameskip": 1},
                           # is_atari=True,
                           clip_rewards=True,
                           normalize_actions=False, 
                           )
config = config.framework(framework="tf")
config = config.rollouts(
                        num_rollout_workers=1,
                        # create_env_on_local_worker=True,
                        num_envs_per_worker=1,
                        rollout_fragment_length=4,
                        preprocessor_pref=None,
                        compress_observations=True
                        )

explore_config = config.exploration_config
explore_config["final_epsilon"] = 0.01
explore_config["epsilon_timesteps"] = 2e5
print(explore_config)
config = config.exploration(
                           explore=True,
                           exploration_config=explore_config
                           )
# config = config.checkpointing(export_native_model_files=True)
config = config.resources(num_gpus=1)

# config = config.evaluation(evaluation_interval=5e5,
#                           evaluation_config=config.training())
# print(config.replay_buffer_config)  

# AIR REFERENCE
# https://docs.ray.io/en/master/ray-air/package-ref.html#components
tuner = tune.Tuner("DQN",
                   run_config=air.RunConfig(stop={"agent_timesteps_total": 5e6},
                                           log_to_file=True,
                                           name="BaselineDQN-Breakout",
                                            # CHECKPOINT REFERENCE
                                            # https://docs.ray.io/en/master/ray-air/package-ref.html#ray.air.config.CheckpointConfig
                                           checkpoint_config=air.CheckpointConfig(num_to_keep=1,
                                                                                  checkpoint_score_attribute="episode_reward_mean",
                                                                                  checkpoint_score_order="max",
                                                                                  checkpoint_frequency=10,
                                                                                  checkpoint_at_end=True,
                                                                                 ),
                                          ),
                   param_space=config.to_dict())

results = tuner.fit()