import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from gymnasium import spaces
import numpy as np


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """
    A custom feature extractor for processing Pokémon battle states.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        self.net = th.nn.Sequential(
            th.nn.Linear(observation_space.shape[0], 64),
            th.nn.ReLU(),
            th.nn.Linear(64, features_dim),
            th.nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)


class RLAgent:
    def __init__(self, env, policy="MlpPolicy"):
        """
        Initialize the RL agent using DQN with a custom environment.
        """
        # Observation space must be a Box for Stable-Baselines3
        self.env = make_vec_env(lambda: env, n_envs=1)
        self.model = DQN(
            policy,
            self.env,
            tensorboard_log="./logs/",
            verbose=1,
            policy_kwargs={
                "features_extractor_class": CustomFeatureExtractor,
                "features_extractor_kwargs": {"features_dim": 128},
            },
        )

    def train(self, total_timesteps=10000):
        """
        Train the RL agent for a specified number of timesteps.
        """
        self.model.learn(total_timesteps=total_timesteps)

    def save_model(self, path="dqn_pokemon"):
        """
        Save the trained model to disk.
        """
        self.model.save(path)

    def load_model(self, path="dqn_pokemon"):
        """
        Load a saved model from disk.
        """
        self.model = DQN.load(path, env=self.env)

    def act(self, observation):
        """
        Get an action based on the current observation.
        """
        action, _states = self.model.predict(observation, deterministic=True)
        return action
