from RLMusicEnv import RLMusicBotEnv
from stable_baselines3 import DQN, PPO
from Agents.rl_agent import RLAgent
from Agents.random_agent import RandomAgent

#Create environment
env = RLMusicBotEnv(bars=8)

# For DQN
args = {
    "target_update_interval": 1000,
    "batch_size": 128,
    "learning_starts": 10000,
    "buffer_size": 200000,
    "exploration_fraction": 0.5,
    "exploration_final_eps": 0.02,
    "train_freq": 4,
    "gradient_steps": 1,
    "gamma": 0.99,
    "learning_rate": 1e-4,
}

#Train agents
# dqn_agent = RLAgent(env, agent_type=DQN, args=args, save_path="./models/dqn_model.zip")
# dqn_agent.train(max_episodes=5000, filename="graphs/dqn_agent.png")

# ppo_agent = RLAgent(env, agent_type=PPO, save_path="./models/ppo_model.zip")
# ppo_agent.train(max_episodes=5000, filename="graphs/ppo_agent.png")

#Test agents
# dqn_agent.load("./models/dqn_model.zip")
# dqn_agent.test("./generated_music/dqn_music.mid", debug=True)

# ppo_agent.load("./models/ppo_model.zip")
# ppo_agent.test("./generated_music/ppo_music.mid", debug=True)

#Random
random_agent = RandomAgent(env)
random_agent.test("./generated_music/random_dqn_music.mid", debug=True)