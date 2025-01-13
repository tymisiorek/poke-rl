from environment.battle_env import create_battle_environment
from agents.rl_agent import RLAgent
import os
import logging

logging.getLogger("poke_env").setLevel(logging.ERROR)


def train_rl_agent():
    # Initialize the environment
    env = create_battle_environment()

    # Initialize the RL agent
    agent = RLAgent(env)

    # Train the agent
    print("Starting training...")
    agent.train(total_timesteps=20000)

    # Save the model
    save_path = "./models/dqn_pokemon"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save_model(save_path)
    print(f"Model saved at {save_path}")


def test_rl_agent():
    # Initialize the environment
    env = create_battle_environment()

    # Load the trained RL agent
    agent = RLAgent(env)
    agent.load_model("./models/dqn_pokemon")

    # Run a sample battle
    obs, _ = env.reset()  # Extract the observation from the tuple
    done = False
    total_reward = 0

    print("Starting a test battle...")
    while not done:
        action = agent.act(obs)  # RL agent action
        obs, reward, done, info = env.step(action)
        obs = obs  # Only pass the observation to the agent
        total_reward += reward

    print(f"Test battle complete! Total reward: {total_reward}")



if __name__ == "__main__":
    # Train the RL agent
    train_rl_agent()

    # Test the RL agent
    test_rl_agent()
