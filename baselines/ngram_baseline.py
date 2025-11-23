import numpy as np
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Laplace
from nltk.lm.vocabulary import Vocabulary

from .baseline_env_utils import note_id_from_action

# Following from: https://www.nltk.org/api/nltk.lm.html

class NGramMusicModel:
    def __init__(self, n=2, vocab_size=None):
        self.ngram_size = n
        self.vocab_size = vocab_size # total span of tokens from Gymnasium env

        self.train = None
        self.tokens = list(map(str, range(self.vocab_size))) + ["<s>", "</s>"] # Possible range from Gymnasium env + the start and stop tokens of NLTK
        self.vocab = Vocabulary(self.tokens)

        self.ngram_model = Laplace(self.ngram_size, vocabulary=self.vocab)

    def fit(self, dataset):
        train, padded_vocab = padded_everygram_pipeline(self.ngram_size, dataset)
        self.train = train

        self.ngram_model.fit(self.train, self.vocab)

    def sample_sequence(self, num_bars=4, random_seed=None, env=None):
        if random_seed is not None:
            np.random.seed(random_seed)

        if env is None:
            raise ValueError("Must be provided a gymnasium environment for sampling")

        composition = []
        action_context = ["<s>"] * (self.ngram_size - 1)

        current_bar = 0
        current_beat = 0.0
        beats_per_bar = env.beats_per_bar

        while current_bar < num_bars:
            remaining_beats = beats_per_bar - current_beat

            # Filter probabilities to take valid actions
            valid_actions = np.zeros(self.vocab_size, dtype=bool)
            action_probs = np.zeros(self.vocab_size)
            for action in range(self.vocab_size):
                _, duration_id, volume_id = note_id_from_action(action, env)
                duration = env.durations[duration_id]

                # Ensure only volume=0.2 (id=2) are selected since that is all learned
                if volume_id != 2:
                    continue

                # Collection all valid actions and their probabilities (could be 0)
                if duration <= remaining_beats:
                    valid_actions[action] = True
                    action_probs[action] = self.ngram_model.score(str(action), action_context)

            # Ensure invalid actions set to 0 probability
            action_probs[~valid_actions] = 0.0

            # Handle no learned transitions for current action_context
            if np.sum(action_probs) == 0:
                possible_actions = valid_actions.astype(float)

                action_probs = possible_actions

            # Normalize to proper probability vector
            action_probs = action_probs / np.sum(action_probs)
            action = int(np.random.choice(self.vocab_size, p=action_probs))

            composition.append(action)
            action_context.pop(0)
            action_context.append(str(action))

            _, action_duration_id, _ = note_id_from_action(action, env)
            action_duration = env.durations[action_duration_id]
            current_beat += action_duration

            if current_beat >= beats_per_bar:
                current_bar += 1
                current_beat = 0.0

        return composition
    