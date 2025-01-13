from environment.battle_env import create_battle_environment
from agents.random_agent import RandomAgent

if __name__ == "__main__":
    # Initialize the environment
    env = create_battle_environment()

    # Initialize a RandomAgent for testing
    agent = RandomAgent()

    # Run a sample battle
    obs = env.reset()
    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)

    print("Sample battle complete!")
