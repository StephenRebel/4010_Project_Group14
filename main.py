from RLMusicEnv import RLMusicBotEnv
from stable_baselines3 import DQN, PPO
from Agents.rl_agent import RLAgent
from Agents.random_agent import RandomAgent

#Create environment
env = RLMusicBotEnv(bars=8)

args = {
    "learning_rate": 1e-4,
    "buffer_size": 1_000_000,
    "learning_starts": 50_000,
    "batch_size": 128,
    "gamma": 0.99,
    "target_update_interval": 8000,
    "train_freq": 4,
    "gradient_steps": 4,
    "exploration_fraction": 0.15,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "tau": 1.0,
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
}

#Train agents
# dqn_agent = RLAgent(env, agent_type=DQN, args=args, save_path="./models/dqn_model.zip")
# dqn_agent.train(max_episodes=20000, filename="graphs/dqn_agent.png")

ppo_agent = RLAgent(env, agent_type=PPO, args=ppo_args, save_path="./models/ppo_model.zip")
ppo_agent.train(max_episodes=10000, filename="graphs/ppo_agent.png")

#Test agents
# dqn_agent.load("./models/dqn_model.zip")
# for i in range(1, 11):
#     dqn_agent.test(midi_filename=f"./generated_music/dqn_music{i}.mid", debug=True, deterministic=False)


ppo_agent.load("./models/ppo_model.zip")
for i in range(1, 11):
    ppo_agent.test(midi_filename=f"./generated_music/ppo_music{i}.mid", debug=True, deterministic=False)

#Random
#random_agent = RandomAgent(env)
# random_agent.test("./generated_music/random_dqn_music.mid", debug=True)