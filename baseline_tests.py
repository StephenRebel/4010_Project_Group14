import json
import torch
import numpy as np

from baselines.ngram_baseline import NGramMusicModel
from baselines.lstm_baseline import LSTMMusicModel

from RLMusicEnv import RLMusicBotEnv
from baselines.baseline_env_utils import actions_to_musical_score, save_score_to_midi, EOS_ID

# Seed if needed
# SEED = 42
SEED = None
N_RUNS = 5000

# Env Parameters
NUM_BARS = 8
VOLUME_ID = 2 # volume=0.8

# Model Parameters
NGRAM_SIZE = 3
VOCAB_SIZE_NGRAM = 320
VOCAB_SIZE_LSTM = 321

# Paths
LSTM_MODEL_PATH = "baselines/models/best_lstm.pth"
DATASET_PATH = "baselines/baseline_dataset_midi.jsonl"

def load_ngram(dataset_path):
    # NLTK expects list of lists for dataset
    baseline_data = []

    with open(dataset_path, "r") as df:
        for line in df:
            json_line = json.loads(line)
            baseline_data.append(list(map(str, json_line["actions"])))

    # Create the model
    ngram_music_model = NGramMusicModel(n=NGRAM_SIZE, vocab_size=VOCAB_SIZE_NGRAM)
    ngram_music_model.fit(baseline_data)

    return ngram_music_model

def test_ngram(env, full_test=False):
    ngram_model = load_ngram(DATASET_PATH)

    if not full_test:
        # Single sample and save as midi
        print("Testing NGram single sequence:\n")

        composition_actions = ngram_model.sample_sequence(num_bars=env.bars, random_seed=SEED, env=env)
        print(f"Generated actions:\n{composition_actions}")

        music_score = actions_to_musical_score(composition_actions, env, None)

        print(f"Generated score:\n{music_score}")
        avg_reward, breakdown = env._compute_reward(music_score)
        print(f"Reward: {avg_reward}\nBreakdown:\n{breakdown}")

        save_score_to_midi("baselines/sample_compositions/ngram_generation_sample_demo.mid", music_score)
    else:
        # Avg reward achieved by the ngram model over N_RUNS runs
        print(f"Testing NGRAM over {N_RUNS} sequences...\n")

        final_rewards = []
        base_rewards = []
        penalties = []

        rhythm_vals = []
        harmony_vals = []
        progression_vals = []
        repetition_vals = []

        for i in range(N_RUNS):
            composition_actions = ngram_model.sample_sequence(num_bars=env.bars, random_seed=SEED, env=env)
            music_score = actions_to_musical_score(composition_actions, env, None)

            final_r, breakdown, base_r = env._compute_reward(music_score)

            final_rewards.append(final_r)
            base_rewards.append(base_r)
            penalties.append(final_r - base_r)

            rhythm_vals.append(breakdown['rhythm'])
            harmony_vals.append(breakdown['harmony'])
            progression_vals.append(breakdown['progression'])
            repetition_vals.append(breakdown['repetition'])

        # averages
        avg_reward = np.mean(final_rewards)
        avg_base = np.mean(base_rewards)
        avg_penalty = np.mean(penalties)

        avg_rhythm = np.mean(rhythm_vals)
        avg_harmony = np.mean(harmony_vals)
        avg_progression = np.mean(progression_vals)
        avg_repetition = np.mean(repetition_vals)

        print(f"Average final reward over {N_RUNS} runs: {avg_reward:.4f}")
        print(f"Average base reward: {avg_base:.4f}")
        print(f"Average penalty (final - base): {avg_penalty:.4f}\n")

        print("Breakdown averages:")
        print(f"\trhythm: {avg_rhythm:.4f}")
        print(f"\tharmony: {avg_harmony:.4f}")
        print(f"\tprogression: {avg_progression:.4f}")
        print(f"\trepetition: {avg_repetition:.4f}")

    return avg_reward

def load_lstm(path, device):
    # Match to training parameters
    model = LSTMMusicModel(
        vocab_size=VOCAB_SIZE_LSTM,
        embed_size=32,
        hidden_size=128,
        layers=1,
        dropout=0.5
    )

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def test_lstm(env, full_test=False):
    from baselines.lstm_baseline import sample_from_lstm

    device = "cuda" if torch.cuda.is_available() else "cpu"

    lstm_model = load_lstm(LSTM_MODEL_PATH, device)

    if not full_test:
        # Single sample and save as midi
        print("Testing LSTM single sequence:\n")

        composition_actions = sample_from_lstm(model=lstm_model, env=env, num_bars=env.bars, device=device, random_seed=SEED)
        print(f"Generated actions:\n{composition_actions}")

        music_score = actions_to_musical_score(composition_actions, env, None)

        print(f"Generated score:\n{music_score}")
        avg_reward, breakdown = env._compute_reward(music_score)
        print(f"Reward: {avg_reward}\nBreakdown:\n{breakdown}")

        save_score_to_midi("baselines/sample_compositions/lstm_generation_sample_demo.mid", music_score)
    else:
        # Avg reward achieved by the ngram model over N_RUNS runs
        print(f"Testing LSTM over {N_RUNS} sequences...\n")

        final_rewards = []
        base_rewards = []
        penalties = []

        rhythm_vals = []
        harmony_vals = []
        progression_vals = []
        repetition_vals = []

        for i in range(N_RUNS):
            composition_actions = sample_from_lstm(model=lstm_model, env=env, num_bars=env.bars, device=device)
            music_score = actions_to_musical_score(composition_actions, env, None)

            final_r, breakdown, base_r = env._compute_reward(music_score)

            final_rewards.append(final_r)
            base_rewards.append(base_r)
            penalties.append(final_r - base_r)

            rhythm_vals.append(breakdown['rhythm'])
            harmony_vals.append(breakdown['harmony'])
            progression_vals.append(breakdown['progression'])
            repetition_vals.append(breakdown['repetition'])

        # averages
        avg_reward = np.mean(final_rewards)
        avg_base = np.mean(base_rewards)
        avg_penalty = np.mean(penalties)

        avg_rhythm = np.mean(rhythm_vals)
        avg_harmony = np.mean(harmony_vals)
        avg_progression = np.mean(progression_vals)
        avg_repetition = np.mean(repetition_vals)

        print(f"Average final reward over {N_RUNS} runs: {avg_reward:.4f}")
        print(f"Average base reward: {avg_base:.4f}")
        print(f"Average penalty (final - base): {avg_penalty:.4f}\n")

        print("Breakdown averages:")
        print(f"\trhythm: {avg_rhythm:.4f}")
        print(f"\tharmony: {avg_harmony:.4f}")
        print(f"\tprogression: {avg_progression:.4f}")
        print(f"\trepetition: {avg_repetition:.4f}")

    return avg_reward

if __name__ == "__main__":
    env = RLMusicBotEnv(bars=NUM_BARS)

    test_ngram(env, full_test=True)

    test_lstm(env, full_test=True)