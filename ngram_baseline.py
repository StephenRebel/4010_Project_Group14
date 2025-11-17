import numpy as np
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Laplace
from nltk.lm.vocabulary import Vocabulary

from baseline_env_utils import note_id_from_action

# Following from: https://www.nltk.org/api/nltk.lm.html

class NGramMusicModel:
    def __init__(self, n, vocab_size):
        self.ngram_size = n
        self.vocab_size = vocab_size # total span of tokens from Gymnasium env

        self.train = None
        self.tokens = list(range(self.vocab_size)) + ["<s>", "</s>"] # Possible range from Gymnasium env + the start and stop tokens of NLTK
        self.vocab = Vocabulary(self.tokens)

        self.ngram_model = Laplace(self.ngram_size, vocabulary=self.vocab)

    def fit(self, dataset):
        train, _ = padded_everygram_pipeline(self.ngram_size, dataset)
        self.train = train

        self.ngram_model.fit(self.train, self.vocab)

    def sample_sequence(self, num_bars=4, random_seed=None, env=None):
        if random_seed is not None:
            np.random.seed(random_seed)

        if env is None:
            raise ValueError("Must be provided a gymnasium environment for sampling")

        composition = [[]]
        action_context = ["<s>"] * (self.ngram_size - 1)

        current_bar = 0
        current_beat = 0.0
        beats_per_bar = 4

        while current_bar < num_bars:
            remaining_beats = beats_per_bar - current_beat

            # Filter probabilities to take valid actions
            valid_actions = np.zeros(self.vocab_size, dtype=bool)
            action_probs = np.zeros(self.vocab_size)
            for action in range(self.vocab_size):
                _, duration_id, _ = note_id_from_action(action, env)
                duration = env.durations[duration_id]

                # Collection all valid actions and their probabilities (could be 0)
                if duration <= remaining_beats:
                    valid_actions[action] = True
                    action_probs[action] = self.ngram_model.score(action, action_context)

            action_probs[~valid_actions] == 0.0

            # Handle no learned transitions for current action_context
            if action_probs.sum() == 0:
                possible_actions = valid_actions.astype(float)

                action_probs = possible_actions

            action_probs = action_probs / action_probs.sum()
            action = np.random.choice(self.vocab_size, p=action_probs)

            composition[current_bar].append(action)
            action_context = action_context[1:] + [action]

            _, action_duration_id, _ = note_id_from_action(action, env)
            action_duration = env.durations[duration_id]
            current_beat += action_duration

            if current_beat >= beats_per_bar:
                current_bar += 1
                current_beat = 0.0
                if current_bar != num_bars:
                    composition.append([])

        return composition
    