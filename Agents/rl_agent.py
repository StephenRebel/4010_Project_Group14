from stable_baselines3 import DQN, PPO
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from .base_agent import CallbackFunction
from collections import Counter

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

    def evaluate(self, n=10, debug=False, deterministic=False, prefix="ppo_music", save_file=False):
        rewards = []
        base_rewards = []
        breakdowns = []
        all_notes = []

        pitches = self.env.pitches      # e.g. [60, 61, ..., 84]
        durations = self.env.durations  # e.g. [0.25, 0.5, 1.0, 2.0]

        print(f"Evaluating agent on {n} generations...")
        print("=" * 90)

        for i in range(1, n + 1):
            midi_file = f"./generated_music/{prefix}{i}.mid" if (save_file and i <= 11) else None
            final_reward, breakdown, base_reward = self.test(
                midi_filename=midi_file,
                debug=debug,
                deterministic=deterministic
            )

            rewards.append(final_reward)
            base_rewards.append(base_reward)
            breakdowns.append(breakdown)

            # Flatten nested score sequences
            for sequence in self.env._musical_score:  # top-level list
                for note in sequence:  # individual (pitch, duration) tuples
                    pitch_val, dur_val = note
                    if dur_val is not None:
                        all_notes.append((pitch_val, dur_val))

            print(f"Piece {i:4d} │ Final: {final_reward:8.4f} │ Base: {base_reward:8.4f} │ "
                f"R:{breakdown['rhythm']:.3f} H:{breakdown['harmony']:.3f} "
                f"P:{breakdown['progression']:.3f} R:{breakdown['repetition']:.3f}")

        rewards = np.array(rewards)
        base_rewards = np.array(base_rewards)

        print("\n" + "=" * 90)
        print(f"EVALUATION COMPLETE — {n} pieces ({prefix.upper()})")
        print("=" * 90)
        print(f"{'Final Reward (with penalty)':<34} → {rewards.mean():.4f} ± {rewards.std():.4f}  [best: {rewards.max():.4f}]")
        print(f"{'Base Reward (pre-penalty)':<34} → {base_rewards.mean():.4f} ± {base_rewards.std():.4f}  [best: {base_rewards.max():.4f}]")
        print(f"{'Average Penalty Applied':<34} → {(base_rewards.mean() - rewards.mean()):.4f}")
        print("-" * 90)

        for comp in ['rhythm', 'harmony', 'progression', 'repetition']:
            vals = [b[comp] for b in breakdowns]
            print(f"{comp.capitalize():12} → {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        if all_notes:
            pitches_in_score = [p for (p, d) in all_notes if isinstance(p, int)]
            num_rests = sum(1 for (p, d) in all_notes if p is None)
            durations_in_score = [d for (p, d) in all_notes]

            pitch_counts = Counter([p for (p, d) in all_notes])
            duration_counts = Counter(durations_in_score)

            pitch_values = [pitch_counts.get(p, 0) for p in pitches]
            duration_values = [duration_counts.get(d, 0) for d in durations]

            if num_rests > 0:
                pitch_values.append(num_rests)
                pitches_with_rests = pitches + ["Rest"]
            else:
                pitches_with_rests = pitches

            total_notes = sum(pitch_values)

            plt.figure(figsize=(14, 5))

            # Pitch histogram
            plt.subplot(1, 2, 1)
            plt.bar(np.arange(len(pitches_with_rests)), pitch_values, color="skyblue", width=0.6)
            plt.xticks(np.arange(len(pitches_with_rests)), pitches_with_rests, rotation=45)
            plt.xlabel("MIDI Pitch")
            plt.ylabel("Count")
            plt.title("Histogram of All Chosen Pitches (Rests included)")

            # Duration histogram
            plt.subplot(1, 2, 2)
            plt.bar(np.arange(len(durations)), duration_values, color="salmon", width=0.6)
            plt.xticks(np.arange(len(durations)), durations, rotation=45)
            plt.xlabel("Duration (beats)")
            plt.ylabel("Count")
            plt.title("Histogram of All Chosen Durations")

            plt.suptitle(f"Evaluation — Note Usage ({n} pieces)", fontsize=14)
            plt.tight_layout()

            os.makedirs("graphs", exist_ok=True)
            hist_filename = f"graphs/{prefix}_evaluation_histogram.png"
            plt.savefig(hist_filename, dpi=200)
            plt.show()
            plt.close()

            print(f"\nHistogram saved → {hist_filename}")
            print(f"   Total notes plotted: {total_notes}")
            if pitches_in_score:
                print(f"   Pitch range used: {min(pitches_in_score)}–{max(pitches_in_score)} (MIDI)")
            else:
                print("   No notes played — only rests present.")
            if num_rests > 0:
                print(f"   Number of rests: {num_rests}")

        else:
            print("\nNo notes generated at all — histogram skipped.")

        print("=" * 90)
        print("Evaluation finished!")
        print("First 11 MIDI files saved in ./generated_music/")
        print("=" * 90)

        return rewards, base_rewards, breakdowns
