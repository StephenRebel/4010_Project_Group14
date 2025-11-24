import os
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

class BaseAgent:
    def __init__(self, env):
        self.env = env
        self.model = None

    def train(self):
        raise NotImplementedError

    def save(self, path):
        self.model.save(path)

    def load(self, path, env=None):
        self.model = self.model.load(path, env=env or self.env)

class CallbackFunction(BaseCallback):
    def __init__(self, max_episodes):
        super().__init__()
        self.max_episodes = max_episodes
        self.episode_rewards = []
        self.current_reward = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        # accumulate reward
        reward = self.locals["rewards"][0]
        self.current_reward += reward

        # check if episode ended
        done = self.locals["dones"][0]
        if done:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0
            self.episode_count += 1

            # stop if we've reached max episodes
            if self.episode_count >= self.max_episodes:
                print(f"Stopping training after {self.max_episodes} episodes")
                return False  # tells SB3 to stop training

        return True
    
    def plot_episode_rewards(ep_rewards, label="Model", smooth=20, filename="graphs/graph.png"):
        rewards = np.array(ep_rewards)

        if len(rewards) >= smooth:
            avg = np.convolve(rewards, np.ones(smooth)/smooth, mode='valid')
            x = np.arange(smooth - 1, len(rewards))
        else:
            avg = rewards
            x = np.arange(len(rewards))

        plt.plot(x, avg, label=label, linewidth=1.5)
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward")
        plt.title("Episode Reward Over Training")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.show()
        plt.close() 