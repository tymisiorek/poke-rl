from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.env_player import Gen8EnvSinglePlayer
from gym import spaces

class BattleEnvironment(Gen8EnvSinglePlayer):
    def __init__(self):
        super().__init__()

    def embed_battle(self, battle: AbstractBattle):
        # Create a custom observation space based on the battle state
        return battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0

    def action_space(self):
        return spaces.Discrete(4)  # Simplified for now (4 moves)

def create_battle_environment():
    return BattleEnvironment()
