import numpy as np
import gymnasium as gym
from collections import Counter
import matplotlib.pyplot as plt


class RLMusicBotEnv(gym.Env):
    def __init__(self, bars: int = 4, render_mode: str = None):
        super(RLMusicBotEnv, self).__init__()

        # Config
        self.bars = bars
        self.beats_per_bar = 4
        self.scale = "C"

        # Note options
        self.pitches = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84]  # 2 octave c major scale
        self.rest_action = len(self.pitches)
        self.n_pitches = len(self.pitches) + 1  # +1 for rest

        self.durations = [0.25, 0.5, 1.0, 2.0, 4.0]
        self.n_durations = len(self.durations)

        self.volumes = [0.4, 0.6, 0.8, 1.0]
        self.n_volumes = len(self.volumes)
        
        # C maj Chord definitions
        self.chords = {
            'C':    [60, 64, 67, 72, 76, 79, 84],   # C E G (C4-C6)
            'Dm':   [62, 65, 69, 74, 77, 81],       # D F A
            'Em':   [64, 67, 71, 76, 79, 83],       # E G B
            'F':    [65, 69, 72, 77, 81, 84],       # F A C
            'G':    [67, 71, 74, 79, 83],           # G B D
            'Am':   [69, 72, 76, 81, 84],           # A C E
            'Bdim': [71, 74, 77, 83],               # B D F
        }

        # Chord progressions weights
        self.progression_weights = {
            # --- C progressions ---
            ('C', 'C'): 0.3,        # repetition (neutral)
            ('C', 'Am'): 1.0,       # I → vi (good)
            ('C', 'Dm'): 1.0,       # I → ii (good)
            ('C', 'Em'): 0.3,       # I → iii (okay)
            ('C', 'F'): 1.0,        # I → IV (good)
            ('C', 'G'): 1.0,        # I → V (good)
            ('C', 'Bdim'): -1.0,    # I → vii° (bad)

            # --- Dm progressions ---
            ('Dm', 'C'): 0.3,       # ii → I (okay)
            ('Dm', 'Dm'): 0.3,      # repetition (neutral)
            ('Dm', 'Em'): -0.5,     # ii → iii (bad)
            ('Dm', 'F'): 0.3,       # ii → IV (okay)
            ('Dm', 'G'): 1.0,       # ii → V (good)
            ('Dm', 'Am'): 1.0,      # ii → vi (good)
            ('Dm', 'Bdim'): -0.5,   # ii → vii° (bad)

            # --- Em progressions ---
            ('Em', 'C'): 0.3,       # iii → I (okay)
            ('Em', 'Dm'): -0.5,     # iii → ii (bad)
            ('Em', 'Em'): 0.3,      # repetition (neutral)
            ('Em', 'F'): 0.3,       # iii → IV (okay)
            ('Em', 'G'): 0.3,       # iii → V (okay)
            ('Em', 'Am'): 0.3,      # iii → vi (okay)
            ('Em', 'Bdim'): -0.5,   # iii → vii° (bad)

            # --- F progressions ---
            ('F', 'C'): 1.0,        # IV → I (good / resolution)
            ('F', 'Dm'): 1.0,       # IV → ii (good)
            ('F', 'Em'): 0.3,       # IV → iii (okay)
            ('F', 'F'): 0.3,        # repetition (neutral)
            ('F', 'G'): 1.0,        # IV → V (good)
            ('F', 'Am'): 0.3,       # IV → vi (okay)
            ('F', 'Bdim'): -1.0,    # IV → vii° (bad)

            # --- G progressions ---
            ('G', 'C'): 1.0,        # V → I (good / strong resolution)
            ('G', 'Dm'): -0.5,      # V → ii (bad)
            ('G', 'Em'): 0.3,       # V → iii (okay)
            ('G', 'F'): 0.3,        # V → IV (okay)
            ('G', 'G'): 0.3,        # repetition (neutral)
            ('G', 'Am'): 0.3,       # V → vi (okay / deceptive cadence)
            ('G', 'Bdim'): -1.0,    # V → vii° (bad)

            # --- Am progressions ---
            ('Am', 'C'): 1.0,       # vi → I (good)
            ('Am', 'Dm'): 1.0,      # vi → ii (good)
            ('Am', 'Em'): -0.5,     # vi → iii (bad)
            ('Am', 'F'): 1.0,       # vi → IV (good)
            ('Am', 'G'): 0.3,       # vi → V (okay)
            ('Am', 'Am'): 0.3,      # repetition (neutral)
            ('Am', 'Bdim'): -0.5,   # vi → vii° (bad)

            # --- Bdim progressions ---
            ('Bdim', 'C'): 1.0,     # vii° → I (good / resolution)
            ('Bdim', 'G'): 1.0,     # vii° → V (good)
            ('Bdim', 'Am'): -1.0,   # vii° → vi (bad)
            ('Bdim', 'Dm'): -1.0,   # vii° → ii (bad)
            ('Bdim', 'Em'): -1.0,   # vii° → iii (bad)
            ('Bdim', 'F'): -1.0,    # vii° → IV (bad)
            ('Bdim', 'Bdim'): 0.3,  # repetition (neutral)
        }

        # Spaces
        self.action_space = gym.spaces.Discrete(self.n_pitches * self.n_durations * self.n_volumes)

        #Define the observation space
        self.MAX_NOTES_PER_BAR = 16
        obs_shape = (self.bars * self.MAX_NOTES_PER_BAR * 3 + 1,)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)


        # Internal state
        self._musical_score = []
        self.current_bar = 0
        self.last_pitch = None
        self.last_duration = 0.0
        self.current_bar_chord = None
        self.prev_bar_chord = None
        self.chord_progression = []  # Store the sequence of chords

        # Rednering
        self.render_mode = render_mode
        self.plot = None
        self.ax = None
        self.clock = 0.5

    def _map_action_to_note(self, action):
        pitch_idx = action // (self.n_durations * self.n_volumes)
        duration_idx = (action % (self.n_durations * self.n_volumes)) // self.n_volumes
        volume_idx = action % self.n_volumes

        pitch = None if pitch_idx == self.rest_action else self.pitches[pitch_idx]
        duration = self.durations[duration_idx]
        volume = self.volumes[volume_idx]

        return (pitch, duration, volume)

    def reset(self, seed=None):
        super().reset(seed=seed)

        self._musical_score = [[] for _ in range(self.bars)]
        self.current_bar = 0
        self.last_pitch = None
        self.last_duration = 0.0
        self.current_bar_chord = None
        self.prev_bar_chord = None
        self.chord_progression = []  # Reset chord progression
        obs = self._get_obs()
        return obs.astype(np.float32), {}
    
    #Get current observation space
    def _get_obs(self):
        # Prepare an empty array for indices
        obs = np.zeros((self.bars, self.MAX_NOTES_PER_BAR, 3), dtype=np.float32)

        # Fill in the notes
        for i, bar in enumerate(self._musical_score[:self.bars]):
            for j, (pitch, duration, volume) in enumerate(bar[:self.MAX_NOTES_PER_BAR]):
                # Convert pitch, duration, volume to indices
                pitch_i = self.pitches.index(pitch) if pitch in self.pitches else self.rest_action
                duration_i = self.durations.index(duration)
                volume_i = self.volumes.index(volume)

                # Normalize to [0,1]
                pitch_norm = pitch_i / (self.n_pitches - 1)
                duration_norm = duration_i / (self.n_durations - 1)
                volume_norm = volume_i / (self.n_volumes - 1)

                obs[i, j] = [pitch_norm, duration_norm, volume_norm]

        obs = obs.flatten()

        #Compute beats left in current bar
        if self.current_bar < self.bars:
            current_bar_notes = self._musical_score[self.current_bar]
            beats_left = max(self.beats_per_bar - sum(n[1] for n in current_bar_notes), 0)
        else:
            beats_left = 0.0

        beats_left_norm = beats_left / self.beats_per_bar
        obs = np.append(obs, [beats_left_norm])

        return obs.flatten()

    def step(self, action):

        #Get note from action space
        note = self._map_action_to_note(action)
        pitch, duration, volume = note

        remaining = self.beats_per_bar - sum([n[1] for n in self._musical_score[self.current_bar]])

        #Punish durations longer than allowed
        if duration > remaining:
            reward = 0
            obs = self._get_obs()
            done = False
            return obs.astype(np.float32), reward, done, False, {}
        
        self._musical_score[self.current_bar].append(note)
        self.last_pitch = pitch
        self.last_duration = duration
        
        # Check if we need to move to next bar
        # If total duration in current bar exceeds beats_per_bar, move to next bar
        total_duration = sum(n[1] for n in self._musical_score[self.current_bar])
        if total_duration >= self.beats_per_bar:
            # Detect the chord for the finished bar
            detected_chord = self._detect_chord(self._musical_score[self.current_bar])
            if detected_chord is not None:
                self.chord_progression.append(detected_chord)
            
            self.prev_bar_chord = self.current_bar_chord
            self.current_bar_chord = detected_chord  # Set the current bar chord
            self.current_bar += 1
            if self.current_bar < self.bars:
                self.current_bar_chord = None

        done = self.current_bar >= self.bars
        
        # For now reward structure 0 unless at terminal state, i.e. only reward
        if done:
            reward = self._compute_reward()
        else:
            reward = 0.0

        obs = self._get_obs()
        info = {'current_chord': self.current_bar_chord}
        #View obs
        #print(f"\nStep Observation (bar x note x [pitch,dur,vol]):\n{obs}")
        return obs.astype(np.float32), reward, done, False, {}

    # reward function components
    # essentially we want to reward
    def _detect_chord(self, bar_notes):
        # Detecting which chord is most prominent in the bar
        # bar_notes is a list of (pitch, duration, volume) tuples
        if not bar_notes:
            return None
        
        # get pitches from notes
        pitches = [note[0] for note in bar_notes if note[0] is not None]   # ignore rests

        # if no pitches, return None
        if not pitches:
            return None
        
        # counting chord scores and finding best match
        chord_scores = {}

        # scoring each chord
        for chord_name, chord_pitches in self.chords.items():
            score = sum(1 for p in pitches if p in chord_pitches)  # simple count of tones in chord
            chord_scores[chord_name] = score  # store the score
        
        # just in case no chord matches
        max_score = max(chord_scores.values())

        # if no chord has any score, return None
        if max_score == 0:
            return None
        
        # return the chord with highest score
        return max(chord_scores, key=chord_scores.get)

    # Function to compute the reward for the current state of the environment
    # Combination of scale adherence, repetition, and rhythm.    
    def _compute_reward(self):
        # Rhythm reward section, may have to look at datasets of MIDI for some of these parameters
        subdivisions = 4 # allowing 16th notes above
        min_note_duration = 0.25

        # Tunable parameters, some maybe from MIDI datasets
        target_syncopation = 0.20 # Syncopation refers to empahsis on notes in offbeat positions
        max_expected_notes_bar = 12.0
        min_expected_notes_bar = 1.0 # Number of notes we expect to see in a typical bar.
        
        max_entropy = np.log(subdivisions)
        target_entropy = 0.7 * max_entropy # Variety of note placement in bar

        notes_played = [] # List of when notes are played
        notes_per_bar = []

        current_sub = 0
        for bar in self._musical_score:
            count_in_bar = 0
            for (pitch, duration, volume) in bar:
                sub_divs_note = int(duration * subdivisions)
                if pitch is not None:
                    notes_played.append(current_sub)
                    count_in_bar += 1
                current_sub += sub_divs_note
            notes_per_bar.append(count_in_bar)

        total_notes_played = len(notes_played)
        # If no notes played that's bad
        if total_notes_played == 0:
            return -1.0

        # Notes played on interger (quarter note) beats, and notes played off intergers beats
        quarter_beat_notes = sum(1 for idx in notes_played if (idx % subdivisions) == 0)
        quarter_beat_ratio = quarter_beat_notes / total_notes_played
        non_qbn_ratio = 1 - quarter_beat_ratio

        # Create a histogram of when notes occur in bars to compute entropy, how simple vs how complex rhythms tend to be.
        note_slots = [int(idx % subdivisions) for idx in notes_played]
        slots_count = Counter(note_slots)
        slot_histogram = np.array([slots_count.get(i, 0) for i in range(subdivisions)], dtype=float)
        slot_histogram = slot_histogram / slot_histogram.sum()

        entropy = -np.sum(slot_histogram * np.log(slot_histogram + 1e-9))
        entropy_score = 1.0 - abs(entropy - target_entropy) / (max_entropy) # Should be 0 to 1

        # https://en.wikipedia.org/wiki/Time_point
        # Interonset interval, time between beginnings of notes and successive notes. Quantifies rhytmic regularity
        if total_notes_played >= 2:
            played_arr = np.array(notes_played, dtype=float)
            ioi = np.diff(played_arr)
            ioi_mean = float(np.mean(ioi))
            ioi_var = float(np.var(ioi))
            epsilon = 1e-8
            ioi_stability = 1.0 - (ioi_var / ((ioi_mean ** 2) * (ioi_var + epsilon))) # should be 0 to 1
        else:
            ioi_stability = 0.5

        # Note density, are we playing a reasonable amount
        avg_notes_per_bar = float(np.mean(notes_per_bar)) if len(notes_per_bar) > 0 else 0.0
        expect_avg_npb = (min_expected_notes_bar + max_expected_notes_bar) / 2
        note_density_score = 1.0 - (abs(avg_notes_per_bar - expect_avg_npb) / (max_expected_notes_bar - min_expected_notes_bar)) # should be 0 to 1

        # Syncopation score, playing notes in off beats
        syncopation_score = 1.0 - (abs(non_qbn_ratio - target_syncopation) / (1.0 + target_syncopation)) # should be 0 to 1

        weights = np.array([1.8, 1.2, 0.9, 0.8, 1.0])
        scores = np.array([quarter_beat_ratio, ioi_stability, entropy_score, note_density_score, syncopation_score])

        rhythm_score = float(np.dot(scores, weights)) / np.sum(weights) # normalized 0 to 1

        # Harmony reward
        harmony_score = 0.0
        total_notes = 0

        for bar_idx, bar in enumerate(self._musical_score):
            chord_name = self._detect_chord(bar)
            if chord_name is None:
                continue

            chord_pitches = self.chords[chord_name]
            next_chord_pitches = []
            if bar_idx + 1 < len(self._musical_score):
                next_chord = self._detect_chord(self._musical_score[bar_idx + 1])
                if next_chord:
                    next_chord_pitches = self.chords[next_chord]

            for (pitch, duration, volume) in bar:
                if pitch is None:
                    continue
                total_notes += 1
                # Scale fit - should always fit since gen is in C Maj
                # if (pitch % 12) in [0, 2, 4, 5, 7, 9, 11]:  # C major pitch classes
                #     harmony_score += 1.0
                # Chord fit - if its in the current bar chord
                if pitch in chord_pitches:
                    harmony_score += 2.0
                # Smooth transition - if it happens to be a transition note that works for both bars
                if next_chord_pitches and pitch in next_chord_pitches:
                    harmony_score += 0.5

        if total_notes > 0:
            harmony_score /= total_notes

        # Normalize to [0, 1] given a perfect harmony score would be 2.5
        harmony_score = harmony_score / 2.5

        # Chord progression reward
        bar_chords = [self._detect_chord(bar) for bar in self._musical_score]

        progression_score = 0.0
        valid_transitions = 0

        for i in range(len(bar_chords) - 1):
            prev_chord = bar_chords[i]
            next_chord = bar_chords[i + 1]
            if prev_chord is None or next_chord is None:
                continue
            valid_transitions += 1
            weight = self.progression_weights.get((prev_chord, next_chord), 0.1)
            progression_score += weight

        # NOTE Already normalized if the one weight 9.0  was a mistake and it was meant to 1.0, other wise come back
        if valid_transitions > 0:
            progression_score /= valid_transitions

        # Repetition reward
        sequence = []
        for bar in self._musical_score:
            for (pitch, duration, volume) in bar:
                # NOTE Do we want to be skipping rests? Some motifs likely include rests as part of the sequence.
                if pitch is not None:  # skip rests
                    sequence.append((pitch, duration))

        repetition_score = 0.0
        if len(sequence) >= 4:
            total_weight = 0.0

            # Check for multiple n-gram sizes (2, 3 ,4)
            # weights slightly favor shorter motifs of 2 or 3 notes
            for n, weight in [(2, 0.4), (3, 0.4), (4, 0.2)]:
                if len(sequence) < n * 2: # sequence has to be long enough for us to check repeats
                    continue

                # Extract all of the length n  subsequences and count the occurences of each n-gram
                ngrams = [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]
                counts = Counter(ngrams)
                # Count how many motifs repeat more than once
                repeated = sum(1 for c in counts.values() if c > 1)
                repetition_ratio = repeated / len(counts)

                # Scaling should be nonlinear as we still should reward smaller repetitions
                score_n = repetition_ratio ** 0.5
                # Weighted sum of the repetition scores across n
                repetition_score += weight * score_n
                total_weight += weight

            if total_weight > 0:
                repetition_score /= total_weight
        else:
            repetition_score = 0.0

        # Ensure all scores are are in proper range by clamping, should already be [0.0, 1.0] but failsafe
        rhythm_norm = float(np.clip(rhythm_score, 0.0, 1.0))
        harmony_norm = float(np.clip(harmony_score, 0.0, 1.0))
        progression_norm = float(np.clip(progression_score, 0.0, 1.0))
        repetition_norm = float(np.clip(repetition_score, 0.0, 1.0))

        # Compute final weight reward score
        final_reward = (
            0.27 * rhythm_norm +
            0.35 * harmony_norm +
            0.28 * progression_norm +
            0.1 * repetition_norm
        )

        return final_reward

    def _init_live_plot(self):
            plt.ion()
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_xlabel("Time (beats)")
            ax.set_ylabel("MIDI Pitch")
            ax.set_title("RL Musicbot Note Generation")
            ax.set_ylim(55, 90)
            ax.set_xlim(0, self.bars * self.beats_per_bar)
            # ax.grid(True, which="both", ls=":", alpha=0.5)
            return fig, ax

    def render(self):
        # Display matplot graph of the generation

        if self.plot is None:
            self.plot, self.ax = self._init_live_plot()

        ax = self.ax
        ax.clear()
        ax.set_xlabel("Time (beats)")
        ax.set_ylabel("MIDI Pitch")
        ax.set_title("RL Musicbot Note Generation")
        ax.set_ylim(55, 90)
        ax.set_xlim(0, self.bars * self.beats_per_bar)
        # ax.grid(True, which="both", ls=":", alpha=0.5) Maybe want to use a grid like bar lines
        # NOTE might want to add small seperation between notes so two quarters don't look like half

        current_time = 0.0

        for bar_notes in self._musical_score:
            for pitch, duration, volume in bar_notes:
                if pitch is not None:
                    # color = plt.cm.viridis(volume)
                    ax.hlines(pitch, current_time, current_time + duration, colors="black", linewidth=4)
                else:
                    ax.hlines(57, current_time, current_time + duration, colors='lightgray', linewidth=2, alpha=0.6)

                current_time += duration

        if self.render_mode == "human":
            ax.figure.canvas.draw_idle()
            ax.figure.canvas.flush_events()
            plt.pause(self.clock)

        return None

    def close(self):
        if self.plot is not None:
            plt.close(self.plot)
            self.plot = None
            self.ax = None

    def save_to_midi(self, filename="generated_music.mid", tempo=120):
        import pretty_midi

        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
        seconds_per_beat = 60.0 / tempo

        current_time = 0.0
        for bar in self._musical_score:
            for (pitch, duration, volume) in bar:
                if pitch is None:
                    # Rest — skip ahead in time
                    current_time += duration * seconds_per_beat
                    continue

                note = pretty_midi.Note(
                    velocity=int(volume * 127),
                    pitch=int(pitch),
                    start=current_time,
                    end=current_time + duration * seconds_per_beat
                )
                instrument.notes.append(note)
                current_time += duration * seconds_per_beat

        pm.instruments.append(instrument)
        pm.write(filename)
        print(f"Saved generated music to {filename}")

# TODO refact beats left calculation as env observation