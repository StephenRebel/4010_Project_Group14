import json

from baselines.ngram_baseline import NGramMusicModel
from RLMusicEnv import RLMusicBotEnv
from baselines.baseline_env_utils import actions_to_musical_score

# Seed if needed
SEED = 42

# Env Parameters
NUM_BARS = 16

env = RLMusicBotEnv(bars=NUM_BARS)

# Model Parameters
NGRAM_SIZE = 3
VOCAB_SIZE = env.action_space.n

# NLTK expects list of lists for dataset
baseline_data = []

with open("baselines/baseline_dataset.jsonl", "r") as df:
    for line in df:
        json_line = json.loads(line)
        baseline_data.append(list(map(str, json_line["composition_actions"])))
print("Processed dataset")

# Create the model
ngram_music_model = NGramMusicModel(n=NGRAM_SIZE, vocab_size=VOCAB_SIZE)
ngram_music_model.fit(baseline_data)

# Generate sequence(s)
composition_actions = ngram_music_model.sample_sequence(num_bars=env.bars, random_seed=SEED, env=env)
print(f"Generated actions:\n{composition_actions}")


music_score = actions_to_musical_score(list(map(int, composition_actions)), env, None)

print(f"Generated score:\n{music_score}")