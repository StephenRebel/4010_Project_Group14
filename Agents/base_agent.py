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
    def __init__(self, max_episodes, pitches, durations):
        super().__init__()
        self.max_episodes = max_episodes
        self.episode_rewards = []
        self.chosen_notes = []
        self.current_reward = 0
        self.episode_count = 0

        self.pitches = pitches
        self.durations = durations
        self.n_pitches = len(pitches)
        self.n_durations = len(durations)

    def _on_step(self) -> bool:
        # accumulate reward
        reward = self.locals["rewards"][0]
        self.current_reward += reward

        action_idx = int(self.locals["actions"][0])
        pitch_idx = action_idx // self.n_durations
        duration_idx = action_idx % self.n_durations
        pitch = self.pitches[pitch_idx-1]
        duration = self.durations[duration_idx-1]

        self.chosen_notes.append((pitch, duration))

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
    
    def plot_episode_rewards(callback, label="Model", smooth=20, filename="graphs/graph.png"):
        rewards = callback.episode_rewards

        #Reward graph
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
        reward_filename = filename.replace(".png", "_reward_graph.png")
        plt.savefig(reward_filename)
        plt.show()
        plt.close() 
