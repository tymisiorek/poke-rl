from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from gymnasium.spaces import Discrete, Box
import numpy as np


class BattleEnvironment(Gen8EnvSinglePlayer):
    def __init__(self, opponent=None):
        """
        Initialize the BattleEnvironment with an opponent.
        """
        # Ensure the opponent is a RandomPlayer if not specified
        if opponent is None:
            opponent = RandomPlayer(battle_format="gen8randombattle")
        super().__init__(opponent=opponent)

        # Define the action space (4 moves for simplicity)
        self.action_space = Discrete(4)

        # Define the observation space: a single float value between 0 and 1
        self.observation_space = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def embed_battle(self, battle: AbstractBattle):
        """
        Creates a custom observation based on the battle state.
        For simplicity, we return the HP fraction of the active Pokémon.
        """
        if battle.active_pokemon:
            return np.array([battle.active_pokemon.current_hp_fraction], dtype=np.float32)
        return np.array([0.0], dtype=np.float32)

    def calc_reward(self, last_battle: AbstractBattle, current_battle: AbstractBattle) -> float:
        """
        Calculate the reward based on the transition from the last battle state to the current battle state.
        This is a basic example that rewards fainting opponent Pokémon and penalizes losing our own Pokémon.
        """
        reward = 0

        # Reward for fainting opponent's Pokémon
        reward += len(current_battle.opponent_team) - sum(
            1 for mon in current_battle.opponent_team.values() if mon.fainted
        )

        # Penalty for losing our Pokémon
        reward -= len(current_battle.team) - sum(
            1 for mon in current_battle.team.values() if mon.fainted
        )

        return reward


    def describe_embedding(self) -> str:
        """
        Describe the embedding used for the battle environment.
        """
        return "Returns a single value: the active Pokémon's HP fraction."


def create_battle_environment():
    """
    Factory function to initialize the custom battle environment.
    """
    opponent = RandomPlayer(battle_format="gen8randombattle")
    return BattleEnvironment(opponent=opponent)
