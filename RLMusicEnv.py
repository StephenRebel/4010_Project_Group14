import numpy as np
import gymnasium as gym
from collections import Counter
import matplotlib.pyplot as plt


class RLMusicBotEnv(gym.Env):
    def __init__(self, bars: int = 4, render_mode: str = None, debug: bool = False):
        super(RLMusicBotEnv, self).__init__()

        # Config
        self.bars = bars
        self.beats_per_bar = 4
        self.scale = "C"
        self.debug = debug

        # Note options
        self.pitches = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84]  # 2 octave c major scale
        self.rest_action = len(self.pitches)
        self.n_pitches = len(self.pitches) + 1  # +1 for rest

        self.durations = [0.25, 0.5, 1.0, 2.0, 4.0]
        self.n_durations = len(self.durations)

        self.volumes = [0.4, 0.6, 0.8, 1.0]
        self.n_volumes = len(self.volumes)
        
        # C maj Chord definitions (triads + sevenths)
        self.chords = {
            'C':      [60, 64, 67, 72, 76, 79, 84],         # C E G (C4-C6)
            'Cmaj7':  [60, 64, 67, 71, 72, 76, 79, 83, 84], # C E G + B
            'Dm':     [62, 65, 69, 74, 77, 81],             # D F A
            'Dm7':    [62, 65, 69, 72, 74, 77, 81, 84],     # D F A + C
            'Em':     [64, 67, 71, 76, 79, 83],             # E G B
            'Em7':    [64, 67, 71, 74, 76, 79, 83],         # E G B + D
            'F':      [65, 69, 72, 77, 81, 84],             # F A C
            'Fmaj7':  [65, 69, 72, 76, 77, 81, 84],         # F A C + E
            'G':      [67, 71, 74, 79, 83],                 # G B D
            'G7':     [65, 67, 71, 74, 77, 79, 83],         # G B D + F
            'Am':     [69, 72, 76, 81, 84],                 # A C E
            'Am7':    [67, 69, 72, 76, 79, 81, 84],         # A C E + G
            'Bdim':   [71, 74, 77, 83],                     # B D F
            'Bø7':    [69, 71, 74, 77, 81, 83],             # B D F + A (half-diminished)
        }

        # Map chords to their functional root (strip 7th/quality variations)
        self.chord_root_map = {
            'C': 'C', 'Cmaj7': 'C',
            'Dm': 'Dm', 'Dm7': 'Dm',
            'Em': 'Em', 'Em7': 'Em',
            'F': 'F', 'Fmaj7': 'F',
            'G': 'G', 'G7': 'G',
            'Am': 'Am', 'Am7': 'Am',
            'Bdim': 'Bdim', 'Bø7': 'Bdim',
        }

        # Chord progressions weights (by functional root)
        # This table applies to both triads and sevenths (e.g., C/Cmaj7, G/G7)
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
        self.MAX_NOTES_PER_BAR = int(self.beats_per_bar / self.durations[0])
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
        
        #For now reward structure 0 unless at terminal state, i.e. only reward
        if done:
            reward, reward_info = self._compute_reward(self._musical_score, debug=self.debug)
        else:
            reward = 0.0
            reward_info = {}

        obs = self._get_obs()
        info = {'current_chord': self.current_bar_chord}
        #View obs
        if self.debug:
            print(f"\nStep Observation (bar x note x [pitch,dur,vol]):\n{obs}")
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
        # weights: beat position (strong beats more important), duration (longer notes are more important)
        # Map of explicit seventh tones per 7th chord (MIDI values across available octaves)
        seventh_tones = {
            'Cmaj7': {71, 83},
            'Dm7':   {72, 84},
            'Em7':   {74},
            'Fmaj7': {76},
            'G7':    {65, 77},
            'Am7':   {67, 79},
            'Bø7':   {69, 81},
        }

        # Chord roots for cadence bias
        # Include chord roots across C4–C6 range (60–84)
        chord_roots = {
            'C': {60, 72, 84},
            'Cmaj7': {60, 72, 84},
            'Dm': {62, 74},
            'Dm7': {62, 74},
            'Em': {64, 76},
            'Em7': {64, 76},
            'F': {65, 77},
            'Fmaj7': {65, 77},
            'G': {67, 79},
            'G7': {67, 79},
            'Am': {69, 81},
            'Am7': {69, 81},
            'Bdim': {71, 83},
            'Bø7': {71, 83},
        }

        for chord_name, chord_pitches in self.chords.items():
            score = 0.0  # score for this chord
            curr_beat = 0.0  # track beat position in bar
            
            # iterate through the notes in the bar
            for (pitch, duration, volume) in bar_notes:
                if pitch is not None and pitch in chord_pitches: # only consider notes in the chord
                    # weighting: strong beats (1 and 3) = 2.0, weak beats (2 and 4) = 1.0
                    beat_position = curr_beat % self.beats_per_bar  # position within the bar
                    is_strong_beat = (beat_position < 0.01) or (abs(beat_position - 2.0) < 0.01)  # check for beat 1 or 3
                    beat_weight = 2.0 if is_strong_beat else 1.0  # strong beats get higher weight
                    
                    # dueation weighting so longer notes matter more
                    duration_weight = duration / 1.0  # normalize by quarter note
                    
                    score += beat_weight * duration_weight  # increment the score
                
                curr_beat += duration  # moving to next beat position
            
            # If evaluating a seventh chord, require its seventh tone presence; otherwise downweight
            if chord_name in seventh_tones:
                has_seventh = any(p in seventh_tones[chord_name] for p in pitches)
                if not has_seventh:
                    score *= 0.4  # downweight seventh chords when the 7th isn't present in melody

            # Downweight Am7 if there is no C present in the bar (functional check)
            if chord_name == 'Am7':
                if not any(p in (72, 84) for p in pitches):  # C across octaves
                    score *= 0.6

            # Heuristic: avoid classifying Em from lone E; require G or B presence
            if chord_name == 'Em':
                has_g_or_b = any(p in (67, 71, 79, 83) for p in pitches)
                if not has_g_or_b:
                    score *= 0.5

            # Heuristic: for G/G7, prefer presence of G or D; if only B present, downweight
            if chord_name in ('G', 'G7'):
                has_g_or_d = any(p in (67, 79, 74) for p in pitches)
                if not has_g_or_d:
                    score *= 0.6
                else:
                    # Boost when G is present (dominant function)
                    has_g = any(p in (67, 79) for p in pitches)
                    if has_g:
                        score += 1.5
                # Extra boost for true dominant when both B and F (tritone) appear
                has_b = any(p in (71, 83) for p in pitches)
                has_f = any(p in (65, 77) for p in pitches)
                if chord_name == 'G7' and (has_b and has_f):
                    score += 1.2

            # Cadence/root emphasis: boost chords whose root matches the final melodic pitch
            if pitches:
                last_pitch = pitches[-1]
                roots = chord_roots.get(chord_name, set())
                if last_pitch in roots:
                    score += 2.0

            # Functional IV emphasis: light boost only when clear triad support present
            if chord_name in ('F', 'Fmaj7'):
                has_f = any(p in (65, 77) for p in pitches)
                if has_f:
                    has_a = any(p in (69, 81) for p in pitches)
                    has_c = any(p in (60, 72, 84) for p in pitches)
                    # Require at least one of A or C alongside F; smaller boost to avoid false positives
                    if has_a or has_c:
                        score += 1.0 if has_a and has_c else 0.4

            chord_scores[chord_name] = score  # storing the  score for this chord
        
        # just in case no chord matches
        max_score = max(chord_scores.values())

        # if no chord has any score, return None
        if max_score == 0:
            return None  # no chord detected
        
        # Special handling to prefer C chord in simple melodies
        c_score = chord_scores.get('C', 0.0)
        
        # Check for strong V (G7) evidence before tonic preference
        # Functional rules for dominant harmony:
        has_g = any(p in (67, 79) for p in pitches)
        has_d = any(p in (62, 74) for p in pitches)
        g7_score = chord_scores.get('G7', 0.0)
        last_pitch = pitches[-1] if pitches else None
        
        # Pattern: D-D-C suggests G7 (D is 5th, C is suspended 4th resolving)
        d_count = sum(1 for p in pitches if p in (62, 74))
        if has_d and d_count >= 2 and last_pitch in (60, 72, 84):
            return 'G7'
        
        # If bar emphasizes G or D, prefer G7
        if has_g and g7_score > c_score:
            return 'G7'
        
        # tonic C chord if close in score
        tonic_threshold = max_score * 0.7  # prefer tonic in simple melodies
        if 'C' in chord_scores and chord_scores['C'] >= tonic_threshold:  # prefer tonic
            return 'C' # return C chord if close enough

        # Prefer dominant G/G7 over Am7 when scores are close (functional harmony bias)
        if 'Am7' in chord_scores:
            dom_score = max(chord_scores.get('G', 0.0), chord_scores.get('G7', 0.0))
            if chord_scores['Am7'] >= tonic_threshold and dom_score >= max_score * 0.6:
                # Choose G7 if seventh (F) present; else G
                has_f = any(p in (65, 77) for p in pitches)
                return 'G7' if has_f else 'G'
        
        # return the chord with highest score, with a safeguard to prefer C over Em when close
        best = max(chord_scores, key=chord_scores.get)
        # If Em wins narrowly over C, prefer tonic C (nursery melody bias)
        if best == 'Em' and 'C' in chord_scores:
            if chord_scores['C'] >= 0.8 * chord_scores['Em']:
                return 'C'
        # Prevent false F classification when no A present and C close in score
        if best in ('F', 'Fmaj7'):
            has_a = any(p in (69, 81) for p in pitches)
            has_c = any(p in (60, 72, 84) for p in pitches)
            if not has_a and has_c and 'C' in chord_scores:
                if chord_scores['C'] >= 0.6 * chord_scores[best]:
                    return 'C'
        # For simple nursery harmonization, label dominant as G7
        if best == 'G':
            return 'G7'
        return best

    # Function to compute the reward for the current state of the environment
    # Combination of scale adherence, repetition, and rhythm.    
    def _compute_reward(self, musical_score, debug=False):
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
        for bar in musical_score:
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
            return 0.0, {
            'rhythm': 0.0,
            'harmony': 0.0,
            'progression': 0.0,
            'repetition': 0.0,
        }

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
        # Compute per-note harmony contributions and detect Diatonic Non-Chord Tones
        # We'll track granular components for debug breakdown.
        harmony_score = 0.0  # final normalized score (computed later)
        total_notes = 0      # melodic notes counted that are not rests
        chord_tone_hits = 0  # number of notes that are chord tones in their bar
        passing_ncts = 0     # number of passing non-chord tones
        neighbor_ncts = 0    # number of neighbor non-chord tones
        escape_ncts = 0      # number of escape tones
        appoggiatura_ncts = 0  # number of appoggiaturas
        anticipation_ncts = 0  # number of anticipations
        suspension_ncts = 0  # number of suspensions
        retardation_ncts = 0  # number of retardations
        pedal_ncts = 0  # number of pedal tones

        # Precompute chord for each bar and build melodic sequence for context
        bar_chords = [self._detect_chord(bar) for bar in musical_score]
        melodic_notes = []  # list of (pitch, bar_idx) in melodic order

        for bar_idx, bar in enumerate(musical_score):
            chord_name = bar_chords[bar_idx]
            # collect melodic notes regardless of chord detection so sequence is continuous
            for (pitch, duration, volume) in bar:
                if pitch is not None:
                    melodic_notes.append((pitch, bar_idx))

            if chord_name is None:
                continue

            chord_pitches = self.chords[chord_name]
            next_chord_pitches = []
            if bar_idx + 1 < len(musical_score):
                next_chord = bar_chords[bar_idx + 1]
                if next_chord:
                    next_chord_pitches = self.chords[next_chord]

            for (pitch, duration, volume) in bar:
                if pitch is None:
                    continue
                total_notes += 1
                # Chord tone: strong boost
                if pitch in chord_pitches:
                    chord_tone_hits += 1
                # Anticipation: note belongs to next bar's chord but NOT current bar's chord (NCT)
                elif next_chord_pitches and pitch in next_chord_pitches:
                    anticipation_ncts += 1

        # Diatonic Non-Chord Tone (NCT) detection and reward 
        # Rules:
        # - Note is NOT a chord tone for its bar
        # - Passing: prev < curr < next or prev > curr > next
        # - Neighbor: prev == next and |curr - prev| == 1 step in self.pitches
        # - Escape: step in one direction, leap (2+ steps) in opposite direction
        nct_bonus = 0.0  # accumulated bonus value (0.5 per passing/neighbor/escape tone)
        for i, (pitch, bar_idx) in enumerate(melodic_notes):
            chord_name = bar_chords[bar_idx]
            if chord_name is None:
                continue
            chord_pitches = self.chords[chord_name]
            # skip chord tones and out-of-scale pitches
            if pitch in chord_pitches:
                continue
            if pitch not in self.pitches:
                continue

            prev_pitch = melodic_notes[i - 1][0] if i - 1 >= 0 else None
            next_pitch = melodic_notes[i + 1][0] if i + 1 < len(melodic_notes) else None

            # Passing tone check
            if prev_pitch is not None and next_pitch is not None:
                if (prev_pitch < pitch < next_pitch) or (prev_pitch > pitch > next_pitch):
                    nct_bonus += 0.5
                    passing_ncts += 1
                    continue

            # Neighbor tone check (prev == next and current is one diatonic step away)
            if prev_pitch is not None and next_pitch is not None and prev_pitch == next_pitch:
                try:
                    idx_curr = self.pitches.index(pitch)
                    idx_prev = self.pitches.index(prev_pitch)
                except ValueError:
                    continue
                if abs(idx_curr - idx_prev) == 1:
                    nct_bonus += 0.5
                    neighbor_ncts += 1
                    continue

            # Escape tone check: step in one direction, leap (2+ steps) in opposite direction
            if prev_pitch is not None and next_pitch is not None:
                try:
                    idx_prev = self.pitches.index(prev_pitch)
                    idx_curr = self.pitches.index(pitch)
                    idx_next = self.pitches.index(next_pitch)
                except ValueError:
                    continue
                # Check if current moves by step from prev (1 step)
                step_from_prev = idx_curr - idx_prev
                # Check if next moves by leap from current (2+ steps) in opposite direction
                leap_to_next = idx_next - idx_curr
                if abs(step_from_prev) == 1 and abs(leap_to_next) >= 2:
                    # Opposite directions: signs must be different
                    if (step_from_prev > 0 and leap_to_next < 0) or (step_from_prev < 0 and leap_to_next > 0):
                        nct_bonus += 0.5
                        escape_ncts += 1
                        continue

            # Appoggiatura check: leap (2+ steps) to NCT, then step (1 step) away in opposite direction
            if prev_pitch is not None and next_pitch is not None:
                try:
                    idx_prev = self.pitches.index(prev_pitch)
                    idx_curr = self.pitches.index(pitch)
                    idx_next = self.pitches.index(next_pitch)
                except ValueError:
                    continue
                # Leap from prev to current (2+ steps)
                leap_from_prev = idx_curr - idx_prev
                # Step from current to next (1 step) in opposite direction
                step_to_next = idx_next - idx_curr
                if abs(leap_from_prev) >= 2 and abs(step_to_next) == 1:
                    # Opposite directions: signs must be different
                    if (leap_from_prev > 0 and step_to_next < 0) or (leap_from_prev < 0 and step_to_next > 0):
                        nct_bonus += 0.5
                        appoggiatura_ncts += 1
                        continue

            # Suspension check: note held from previous bar's chord, not in current chord, resolves down by step
            if bar_idx > 0 and prev_pitch is not None and next_pitch is not None:
                prev_bar_chord = bar_chords[bar_idx - 1]
                if prev_bar_chord:
                    prev_bar_chord_pitches = self.chords[prev_bar_chord]
                    # Check if current pitch belongs to previous bar's chord
                    if pitch in prev_bar_chord_pitches:
                        # Check if it resolves down by step
                        try:
                            idx_curr = self.pitches.index(pitch)
                            idx_next = self.pitches.index(next_pitch)
                        except ValueError:
                            continue
                        step_to_next = idx_next - idx_curr
                        # Suspension resolves downward by step (step_to_next == -1)
                        if step_to_next == -1:
                            nct_bonus += 0.5
                            suspension_ncts += 1
                            continue
                        # Retardation resolves upward by step (step_to_next == +1)
                        if step_to_next == 1:
                            nct_bonus += 0.5
                            retardation_ncts += 1
                            continue

            # Pedal tone check: repeated pitch (prev == curr) that doesn't fit current chord
            # but was likely a chord tone in a different harmonic context
            if prev_pitch is not None and prev_pitch == pitch:
                # It's a repeated note - check if it fits any nearby chord context
                # If it doesn't fit current bar but does fit prev or next bar, it's acting as a pedal
                fits_prev_bar = False
                if bar_idx > 0:
                    prev_bar_chord = bar_chords[bar_idx - 1]
                    if prev_bar_chord:
                        fits_prev_bar = pitch in self.chords[prev_bar_chord]
                
                fits_next_bar = False
                if bar_idx + 1 < len(bar_chords):
                    next_bar_chord = bar_chords[bar_idx + 1]
                    if next_bar_chord:
                        fits_next_bar = pitch in self.chords[next_bar_chord]
                
                # Pedal tone: repeated note that fit a different chord context
                if fits_prev_bar or fits_next_bar:
                    nct_bonus += 0.5
                    pedal_ncts += 1
                    continue

        raw_harmony_points = (
            chord_tone_hits * 2.0 +
            passing_ncts * 0.5 +
            neighbor_ncts * 0.5 +
            escape_ncts * 0.5 +
            appoggiatura_ncts * 0.75 +
            anticipation_ncts * 0.6 +
            suspension_ncts * 0.75 +
            retardation_ncts * 0.65 +
            pedal_ncts * 0.6
        )
        per_note_average = (raw_harmony_points / total_notes) if total_notes > 0 else 0.0
        # Normalize to [0, 1] given a perfect per-note value would be 1.7 (sweet spot between 1.5 and 2.0)
        harmony_score = per_note_average / 1.7

        # Chord progression reward
        bar_chords = [self._detect_chord(bar) for bar in musical_score]

        progression_score = 0.0
        valid_transitions = 0

        for i in range(len(bar_chords) - 1):
            prev_chord = bar_chords[i]
            next_chord = bar_chords[i + 1]
            if prev_chord is None or next_chord is None:
                continue
            valid_transitions += 1
            # Map to functional roots (C/Cmaj7 → C, G/G7 → G, etc.)
            prev_root = self.chord_root_map.get(prev_chord, prev_chord)
            next_root = self.chord_root_map.get(next_chord, next_chord)
            weight = self.progression_weights.get((prev_root, next_root), 0.1)
            progression_score += weight

        # NOTE Already normalized if the one weight 9.0  was a mistake and it was meant to 1.0, other wise come back
        if valid_transitions > 0:
            progression_score /= valid_transitions

        # Repetition reward
        # 1. Motif repetition (melodic n-grams including rests)
        sequence = []
        for bar in musical_score:
            for (pitch, duration, volume) in bar:
                # Include rests (None) to recognize rest patterns in motifs
                sequence.append(pitch)

        repetition_motif_score = 0.0
        if len(sequence) >= 2:
            total_weight = 0.0
            for n, weight in [(1, 0.3), (2, 0.4), (3, 0.3)]:
                if len(sequence) < n:
                    continue
                ngrams = [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]
                if len(ngrams) == 0:
                    continue
                counts = Counter(ngrams)
                repeated = sum(1 for c in counts.values() if c > 1)
                repetition_ratio = repeated / len(counts) if len(counts) > 0 else 0
                score_n = repetition_ratio ** 0.5
                repetition_motif_score += weight * score_n
                total_weight += weight
            if total_weight > 0:
                repetition_motif_score /= total_weight
        else:
            repetition_motif_score = 0.0

        # 2. Bar repetition (identical bars)
        bar_tuples = [tuple(bar) for bar in musical_score]
        bar_counts = Counter(bar_tuples)
        repeated_bars = sum(1 for c in bar_counts.values() if c > 1)
        bar_repetition_ratio = repeated_bars / len(bar_counts) if len(bar_counts) > 0 else 0
        repetition_bar_score = bar_repetition_ratio ** 0.5

        # 3. Rhythm repetition: proportion of rhythm patterns that repeat (similar to bar repetition logic)
        rhythm_tuples = [tuple([note[1] for note in bar]) for bar in musical_score]
        rhythm_counts = Counter(rhythm_tuples)
        if len(rhythm_counts) > 0:
            repeated_rhythms = sum(1 for c in rhythm_counts.values() if c > 1)
            rhythm_repetition_ratio = repeated_rhythms / len(rhythm_counts) if len(rhythm_counts) > 0 else 0
            repetition_rhythm_score = rhythm_repetition_ratio ** 0.5
        else:
            repetition_rhythm_score = 0.0

        # Combine all repetition components (equal weights)
        repetition_score = (repetition_motif_score + repetition_bar_score + repetition_rhythm_score) / 3.0

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

        if self.debug:
            # Get detected chords for display
            detected_chords = [self._detect_chord(bar) for bar in musical_score]
            chords_str = ", ".join([str(c) if c else "None" for c in detected_chords])

            print("\n" + "="*60)
            print("REWARD")
            print("="*60)
            print(f"Overall: {final_reward:.4f}")
            # Rhythm detailed breakdown
            print(f"  Rhythm:      {rhythm_norm:.4f}")
            print(f"    - Quarter beat ratio:    {quarter_beat_ratio:.4f} (weight: 1.8)")
            print(f"    - IOI stability:         {ioi_stability:.4f} (weight: 1.2)")
            print(f"    - Entropy score:         {entropy_score:.4f} (weight: 0.9)")
            print(f"    - Note density:          {note_density_score:.4f} (weight: 0.8)")
            print(f"    - Syncopation:           {syncopation_score:.4f} (weight: 1.0)")
            print(f"    - Total notes played:    {total_notes_played}")
            print(f"    - Avg notes per bar:     {avg_notes_per_bar:.2f} (target: {expect_avg_npb:.1f})")
            # Harmony detailed breakdown
            print(f"  Harmony:     {harmony_norm:.4f}")
            print(f"    - Total melodic notes:   {total_notes}")
            print(f"    - Chord tones:           {chord_tone_hits} (x2.0 = {chord_tone_hits * 2.0:.1f})")
            print(f"    - Passing NCTs:          {passing_ncts} (x0.5 = {passing_ncts * 0.5:.1f})")
            print(f"    - Neighbor NCTs:         {neighbor_ncts} (x0.5 = {neighbor_ncts * 0.5:.1f})")
            print(f"    - Escape NCTs:           {escape_ncts} (x0.5 = {escape_ncts * 0.5:.1f})")
            print(f"    - Appoggiatura NCTs:     {appoggiatura_ncts} (x0.75 = {appoggiatura_ncts * 0.75:.1f})")
            print(f"    - Anticipation NCTs:     {anticipation_ncts} (x0.6 = {anticipation_ncts * 0.6:.1f})")
            print(f"    - Suspension NCTs:       {suspension_ncts} (x0.75 = {suspension_ncts * 0.75:.1f})")
            print(f"    - Retardation NCTs:      {retardation_ncts} (x0.65 = {retardation_ncts * 0.65:.1f})")
            print(f"    - Pedal Tone NCTs:       {pedal_ncts} (x0.6 = {pedal_ncts * 0.6:.1f})")
            print(f"    - Raw harmony points:    {raw_harmony_points:.2f}")
            print(f"    - Per-note average:      {per_note_average:.4f}")
            print(f"    - Normalized (/1.7):     {harmony_norm:.4f}")
            # Progression section
            print(f"  Progression: {progression_norm:.4f}")
            print(f"    - Chords detected:       [{chords_str}]")
            # Chord progression transitions
            for i in range(len(detected_chords) - 1):
                c1 = detected_chords[i]
                c2 = detected_chords[i + 1]
                if c1 is not None and c2 is not None:
                    root1 = self.chord_root_map.get(c1, c1)
                    root2 = self.chord_root_map.get(c2, c2)
                    weight = self.progression_weights.get((root1, root2), 0.1)
                    print(f"    - Transition {i+1:4d}->{i+2:1d}:    {c1} -> {c2} (weight: {weight:.1f})")
            # Repetition breakdown
            print(f"  Repetition:  {repetition_norm:.4f}")
            print(f"    - Motif:   {repetition_motif_score:.4f}")
            print(f"    - Bar:     {repetition_bar_score:.4f}")
            print(f"    - Rhythm:  {repetition_rhythm_score:.4f}")
            print("="*60 + "\n")

        # Return both final reward and breakdown dictionary
        return final_reward, {
            'rhythm': rhythm_norm,
            'harmony': harmony_norm,
            'progression': progression_norm,
            'repetition': repetition_norm,
        }

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