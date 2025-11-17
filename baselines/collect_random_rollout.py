import random
from RLMusicEnv import RLMusicBotEnv
from baselines.baseline_env_utils import musical_score_to_actions
from tqdm import tqdm
import json

# Collect a number of msucial compositions to use for training baseline models
def collect_sequences(env, n_compositions=5000, mix=0.6, seed=42):
    random.seed(seed)
    sequences = []

    for comp in tqdm(range(n_compositions)):
        obs, _ = env.reset()
        done = False

        # Choose policy for generation, random or heuristic based
        use_heuristic = random.random() < mix

        while not done:
            if use_heuristic:
                # Simple heuristic of quarter notes and lower half of scale
                pitch_id = random.choice(range(len(env.pitches) // 2))
                duration_id = 2
                volume_id = random.randrange(env.n_volumes)

                mult = env.n_durations * env.n_volumes
                action = pitch_id * mult + duration_id * env.n_volumes + volume_id
            else:
                # Random notes
                action = env.action_space.sample()
            
            obs, reward, done, _, _ = env.step(int(action))
        sequences.append(musical_score_to_actions(env._musical_score, env))

    return sequences

env = RLMusicBotEnv(bars=16)
sequences = collect_sequences(env, n_compositions=5000, mix=0.5)
with open("baseline_dataset.jsonl", "w") as f:
    for id, composition in enumerate(sequences):
        f.write(json.dumps({ "id": id, "composition_actions": composition }) + "\n")