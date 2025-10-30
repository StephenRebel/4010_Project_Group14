import numpy as np
import gymnasium as gym
from collections import Counter

class RLMusicBotEnv(gym.Env):
    def __init__(self, bars: int = 4, render_mode: str = None):
        super(RLMusicBotEnv, self).__init__()

        # Config
        self.bars = bars
        self.beats_per_bar = 4
        self.scale = "C"

        # Note options
        self.pitches = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83]  # 2 octave c major scale
        self.rest_action = len(self.pitches)
        self.n_pitches = len(self.pitches) + 1  # +1 for rest

        self.durations = [0.25, 0.5, 1.0, 2.0, 4.0]
        self.n_durations = len(self.durations)

        self.volumes = [0.4, 0.6, 0.8, 1.0]
        self.n_volumes = len(self.volumes)
        
        # Chord definitions
        self.chords = {
            'C': [60, 64, 67, 72, 76, 79],  # C, E, G
            'F': [65, 69, 60, 77, 81, 72],  # F, A, C 
            'G': [67, 71, 62, 79, 83, 74]   # G, B, D
        }

        # Chord progressions weights
        # MOST Common progression - C F G C
        self.progression_weights = {
            ('C', 'C'): 0.3,  # stay on C
            ('C', 'F'): 1.0,  # C to F (strong)
            ('C', 'G'): 9.0,  # C to G (strong)
            ('F', 'C'): 0.8,  # F to C
            ('F', 'G'): 1.0,  # F to G
            ('F', 'F'): 0.3,  # stay on F
            ('G', 'C'): 1.0,  # G to C (resolution)
            ('G', 'F'): 0.7,  # G to F
            ('G', 'G'): 0.3,  # stay on G
        }

        # Spaces
        self.action_space = gym.spaces.MultiDiscrete([
            self.n_pitches,
            self.n_durations,
            self.n_volumes
        ])

        #Define the observation space: An np array of bars (array) of notes (array), contains indices NOT values
        self.MAX_NOTES_PER_BAR = 16
        highs = [self.n_pitches-1, self.n_durations-1, self.n_volumes-1]
        self.observation_space = gym.spaces.MultiDiscrete([highs] * self.MAX_NOTES_PER_BAR * bars)

        # Internal state
        self._musical_score = []
        self.current_bar = 0
        self.last_pitch = None
        self.last_duration = 0.0
        self.current_bar_chord = None
        self.prev_bar_chord = None

        # Rednering
        self.render_mode = render_mode
        self.plot = None
        self.ax = None
        self.clock = 0.5

    def _map_action_to_note(self, action):
        pitch_idx, duration_idx, volume_idx = action
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
        obs = self._get_obs()

        self.close()

        return obs, {}
    
    #Get current observation space
    def _get_obs(self):
        # Prepare an empty array of indices
        obs = np.zeros((self.bars, self.MAX_NOTES_PER_BAR, 3), dtype=np.int32)
        
        # Iterate over bars
        for i, bar in enumerate(self._musical_score[:self.bars]):
            for j, (pitch, duration, volume) in enumerate(bar[:self.MAX_NOTES_PER_BAR]):
                # Convert pitch, duration, volume to indices
                pitch_i = self.pitches.index(pitch) if pitch in self.pitches else self.rest_action
                duration_i = self.durations.index(duration)
                volume_i = self.volumes.index(volume)
                
                obs[i, j] = [pitch_i, duration_i, volume_i]
        
        return obs
    
    # reward function components
    # essentially we want to reward
    def _detect_chord(self, bar_notes):
        # Detecting which chord (C, F, or G) is most prominent in the bar
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

    def step(self, action):

        #Get note from action space
        note = self._map_action_to_note(action)
        pitch, duration, volume = note
        
        self._musical_score[self.current_bar].append(note)
        self.last_pitch = pitch
        self.last_duration = duration
        
        # Check if we need to move to next bar
        # If total duration in current bar exceeds beats_per_bar, move to next bar
        total_duration = sum(n[1] for n in self._musical_score[self.current_bar])
        if total_duration >= self.beats_per_bar:
            self.prev_bar_chord = self.current_bar_chord
            self.current_bar += 1
            if self.current_bar < self.bars:
                self.current_bar_chord = None

        done = self.current_bar >= self.bars
        
        # For now reward structure 0 unless at terminal state, i.e. only reward
        if done:
            reward = self._compute_reward()
        else:
            reward = 0.0

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        info = {'current_chord': self.current_bar_chord}
        #View obs
        #print(f"\nStep Observation (bar x note x [pitch,dur,vol]):\n{obs}")
        return obs, reward, done, False, info

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
            ioi_stability = 1.0 - (ioi_var / (ioi_mean ** 2 * ioi_var)) # should be 0 to 1
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

        rhythm_reward = float(np.dot(scores, weights))

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

        if valid_transitions > 0:
            progression_score /= valid_transitions

        # Repetition reward
        sequence = []
        for bar in self._musical_score:
            for (pitch, duration, volume) in bar:
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

        final_reward = (
            0.27 * rhythm_reward +
            0.35 * harmony_score +
            0.28 * progression_score +
            0.1 * repetition_score
        )

        return final_reward

    def _init_live_plot(self):
            import matplotlib.pyplot as plt

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
        import matplotlib.pyplot as plt

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

                plt.draw()

                current_time += duration

        if self.render_mode == "human":
            plt.show()
            plt.pause(self.clock)

        return None

    def close(self):
        if self.plot is not None:
            import matplotlib.pyplot as plt
            plt.close(self.plot)

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