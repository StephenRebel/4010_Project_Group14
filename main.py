from RLMusicEnv import RLMusicBotEnv
from stable_baselines3 import DQN, PPO
from Agents.rl_agent import RLAgent
from Agents.random_agent import RandomAgent
from collections import defaultdict

#Create environment
env = RLMusicBotEnv(bars=8)

dqn_args = {
    "learning_rate": 5e-5,
    "gamma": 0.99,
    "buffer_size": 250000,
    "target_update_interval": 5000,
    "batch_size": 128,
    "exploration_fraction": 0.5,
    "batch_size": 128,
}

ppo_args = {
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.015,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "seed": 5,
}

# #Train agents
# dqn_agent = RLAgent(env, agent_type=DQN, args=dqn_args, save_path="./models/dqn_model.zip")
# dqn_agent.train(max_episodes=10000, filename="graphs/dqn_agent.png")

ppo_agent = RLAgent(env, agent_type=PPO, args=ppo_args, save_path="./models/ppo_model.zip")
ppo_agent.train(max_episodes=10000, filename="graphs/ppo_agent.png")

#Test agents
# dqn_agent.load("./models/dqn_model.zip")
# dqn_agent.evaluate(n=5000, debug=False, deterministic=False, prefix='dqn_music', save_file=True)

ppo_agent.load("./models/ppo_model.zip")
ppo_agent.evaluate(n=5000, debug=True, deterministic=False, prefix='ppo_music', save_file=True)

#Random
#random_agent = RandomAgent(env)
#random_agent.test("./generated_music/random_dqn_music.mid", debug=True)