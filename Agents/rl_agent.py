from stable_baselines3 import DQN, PPO
import os
import numpy as np
from collections import defaultdict
from .base_agent import CallbackFunction

class RLAgent:
    def __init__(self, env, agent_type=DQN, args=None, save_path="./models/rl_model.zip"):
        self.env = env
        self.save_path = save_path
        if args is None:
            args = {}
        self.model = agent_type("MlpPolicy", env, verbose=1, **args)

    def train(self, total_timesteps=int(1e9), max_episodes=10000, filename=""):
        callback = CallbackFunction(max_episodes=max_episodes, pitches=self.env.pitches, durations=self.env.durations)  
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        CallbackFunction.plot_episode_rewards(callback, label="Model", smooth=20, filename=filename)
        self.save(self.save_path)

    def test(self, render=True, midi_filename=None, debug=False, deterministic=False):
        obs, _ = self.env.reset()
        done = False
        final_reward = 0
        while not done:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, done, _, _info = self.env.step(action)
            if render and self.env.render_mode == "human":
                self.env.render()

        final_reward, breakdown, base_reward = self.env._compute_reward(self.env._musical_score, debug=debug)

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

        return final_reward, breakdown, base_reward

    def save(self, path):
        self.model.save(path)
        print(f"Saved model to {path}")

    def load(self, path):
        self.model.load(path)
        print(f"Loaded model from {path}")

    def evaluate(self, n=10, debug=False, deterministic=False, prefix="dqn_music", save_file=False):
            
            rewards = []
            base_rewards = []
            breakdowns = []

            print(f"Evaluating agent on {n} generations...")
            print("=" * 90)

            for i in range(1, n + 1):
                midi_file = f"./generated_music/{prefix}{i}.mid"
                if save_file == False or i > 11:
                    midi_file = None

                final_reward, breakdown, base_reward = self.test(
                    midi_filename=midi_file,
                    debug=debug,
                    deterministic=deterministic
                )

                rewards.append(final_reward)
                base_rewards.append(base_reward)
                breakdowns.append(breakdown)

                print(f"Piece {i:2d} │ Final: {final_reward:.4f} │ "
                    f"R:{breakdown['rhythm']:.3f} H:{breakdown['harmony']:.3f} "
                    f"P:{breakdown['progression']:.3f} R:{breakdown['repetition']:.3f}")

            print("=" * 90)

            # Convert to numpy for easy stats
            rewards = np.array(rewards)
            base_rewards = np.array(base_rewards)

            print(f"\nFINAL EVALUATION — {len(rewards)} generations")
            print("=" * 90)
            print(f"{'TOTAL REWARD (final)':<28} → {rewards.mean():.4f} ± {rewards.std():.4f}   [best: {rewards.max():.4f}]")
            print(f"{'BASE REWARD (pre-penalty)':<28} → {base_rewards.mean():.4f} ± {base_rewards.std():.4f}")
            print("-" * 90)

            components = ['rhythm', 'harmony', 'progression', 'repetition']
            for comp in components:
                vals = [b[comp] for b in breakdowns]
                print(f"{comp.capitalize():12} → {np.mean(vals):.4f} ± {np.std(vals):.4f}   "
                    f"(range: {min(vals):.3f}–{max(vals):.3f})")

            print("=" * 90)
            print("All MIDI files saved in ./generated_music/")
            print("Evaluation complete!")
