import random

class RandomAgent:
    def __init__(self):
        pass

    def act(self, observation):
        # Randomly choose an action (0-3)
        return random.choice(range(4))
