from collections import defaultdict, Counter
import numpy as np
from baseline_env_utils import note_id_from_action

class NGramMusicModel:
    def __init__(self, n, vocab_size):
        self.n = n
        self.vocab_size = vocab_size
        self.counts = defaultdict(Counter)

    def fit(self, sequences):
        for sequence in sequences:
            note_context = [-1] * (self.n - 1)
            for a in sequence:
                self.counts[tuple(note_context)][a] += 1
                note_context = (note_context + [a])[-(self.n - 1):]

    def get_probs(self, note_context):
        freq_counter = self.counts.get(tuple(note_context), None)
        if freq_counter is None:
            return np.ones(self.vocab_size) / float(self.vocab_size)
        freqs = np.array([freq_counter.get(i, 0) + 1 for i in range(self.vocab_size)], dtype=float)

        return freqs / freqs.sum()
    
    # Guarentee the same format ensured by the gymnasium environment. More difficult to capture with external models
    def sample_sequence(self, start_ctx=None, eos_id=None, env=None):
        if env is None:
            raise ValueError("Must be provided a gymnasium environment for sampling")

        if start_ctx is None:
            note_context = [-1] * (self.n - 1)
        else:
            note_context = start_ctx[-(self.n - 1):]

        # Tracking state vars
        out = []
        current_bar = 0
        current_beats = 0.0
        beats_per_bar = env.beats_per_bar
        num_bars = env.bars

        # NOTE may need to consider hard limit but should end
        while current_bar < num_bars:
            probs = self.get_probs(note_context).copy()

            remaining_beats = beats_per_bar - current_beats

            # Main filtering loop
            for i in range(self.vocab_size):
                if i == eos_id:
                    # Eos only for the end
                    if not (current_bar == num_bars - 1 and remaining_beats == 0):
                        probs[i] = 0
                else:
                    _, duration_id, _ = note_id_from_action(i, env)
                    duration = env.durations[duration_id]

                    # Filter valid actions, or moving to next bar
                    if remaining_beats > 0:
                        if duration > remaining_beats:
                            probs[i] = 0
                    else:
                        if current_bar == num_bars - 1:
                            probs[i] = 0

            # Handle no valid actions in learned set
            if probs.sum() == 0:
                if current_bar == num_bars - 1 and remaining_beats == 0:
                    if eos_id is not None:
                        out.append(eos_id)
                    break
                else:
                    # Fall back to force a rest with suitable duration
                    forced = []
                    for i in range(self.vocab_size):
                        pitch_id, duration_id, _ = note_id_from_action(i, env)
                        duration = env.durations[duration_id]
                        if pitch_id == env.rest_action and (remaining_beats == 0 or duration <= remaining_beats):
                            forced.append(i)
                    if forced:
                        prob_val = 1.0 / len(forced)
                        for i in forced:
                            probs[i] = prob_val
                    else:
                        # Uniformly random action worst case (ignores masking)
                        probs = np.ones(self.vocab_size) / self.vocab_size

            # Final normalization of probabilities
            if probs.sum() > 0:
                probs /= probs.sum()

            next_action = np.random.choice(range(self.vocab_size), p=probs)
            out.append(int(next_action))

            if eos_id is not None and next_action == eos_id:
                break

            note_context = (note_context + [next_action])[-(self.n - 1):]

            # Track state for filling out simulated musical score
            _, duration_id, _ = note_id_from_action(next_action, env)
            duration = env.durations[duration_id]
            if current_beats + duration > env.beats_per_bar:
                current_bar += 1
                current_beats = 0.0
            current_beats += duration

        return out