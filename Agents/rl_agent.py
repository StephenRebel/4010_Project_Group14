from stable_baselines3 import DQN, PPO
import os
from .base_agent import CallbackFunction

class RLAgent:
    def __init__(self, env, agent_type=DQN, args=None, save_path="./models/rl_model.zip"):
        self.env = env
        self.save_path = save_path
        if args is None:
            args = {}
        self.model = agent_type("MlpPolicy", env, verbose=1, **args)

    def train(self, total_timesteps=int(1e6), max_episodes=10000, filename=""):
        callback = CallbackFunction(max_episodes=max_episodes)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        episode_rewards = callback.episode_rewards
        CallbackFunction.plot_episode_rewards(episode_rewards, label="Model", smooth=20, filename=filename)
        self.save(self.save_path)

    def test(self, render=True, midi_filename=None, debug=False):
        obs, _ = self.env.reset()
        done = False
        final_reward = 0
        while not done:
            action, _ = self.model.predict(obs)
            obs, reward, done, _, _info = self.env.step(action)
            if render and self.env.render_mode == "human":
                self.env.render()

        final_reward = self.env._compute_reward(self.env._musical_score, debug=debug)

        print("\nTest completed.")
        print("Musical score:")
        print(self.env._musical_score)
        print("Chord progression:")
        print(", ".join(self.env.chord_progression))
        print("Reward:")
        print(final_reward)

        if midi_filename:
            os.makedirs(os.path.dirname(midi_filename), exist_ok=True)
            self.env.save_to_midi(midi_filename)

    def save(self, path):
        self.model.save(path)
        print(f"Saved model to {path}")

    def load(self, path):
        self.model.load(path)
        print(f"Loaded model from {path}")
