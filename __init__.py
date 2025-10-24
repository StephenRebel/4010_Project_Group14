import numpy as np
import gymnasium as gym
import pretty_midi
import matplotlib.pyplot as plt
from music21 import key as m21key, pitch as m21pitch
from music21 import stream, note, meter, tempo
import threading
from collections import Counter


class RLMusicBotEnv(gym.Env):
    def __init__(self, bars: int = 4):
        super(RLMusicBotEnv, self).__init__()

        # Config
        self.bars = bars
        self.beats_per_bar = 4
        self.scale = "C"

        # Note options
        self.durations = [0.25, 0.5, 1.0, 2.0, 4.0]
        self.n_durations = len(self.durations)

        self.volumes = [0.4, 0.6, 0.8, 1.0]
        self.n_volumes = len(self.volumes)

        self.pitches = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83]  # 2 octave c major scale
        self.rest_action = len(self.pitches)
        self.n_pitches = len(self.pitches) + 1  # +1 for rest

        # Spaces
        self.action_space = gym.spaces.MultiDiscrete([
            self.n_pitches,
            self.n_durations,
            self.n_volumes
        ])
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Internal state
        self._musical_score = []
        self.current_bar = 0

    def _map_action_to_note(self, action):
        pitch_idx, duration_idx, volume_idx = action
        pitch = None if pitch_idx == self.rest_action else self.pitches[pitch_idx]
        duration = self.durations[duration_idx]
        volume = self.volumes[volume_idx]
        return (pitch, duration, volume)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._musical_score = [[] for _ in range(self.bars)]
        self.current_bar = 0
        state = np.zeros((1,), dtype=np.float32)
        return state, {}
    
    def _get_obs(self):
        return np.array([self._musical_score])

    def step(self, action):
        note = self._map_action_to_note(action)
        self._musical_score[self.current_bar].append(note)

        total_duration = sum(n[1] for n in self._musical_score[self.current_bar] if n[0] is not None)
        if total_duration >= self.beats_per_bar:
            self.current_bar += 1

        done = self.current_bar >= self.bars
        
        # For now reward structure 0 unless at terminal state, i.e. only reward
        if done:
            reward = self._compute_reward()
        else:
            reward = 0.0

        state = np.zeros((1,), dtype=np.float32)
        info = {}
        return state, reward, done, False, info

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

        return (1/3) * rhythm_reward # TODO weighted average of all our reward signals

    def save_to_midi(self, filename="generated_music.mid", tempo=120):
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

def init_live_plot():
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel("Time (beats)")
    ax.set_ylabel("MIDI Pitch")
    ax.set_title("RL Musicbot Note Generation")
    ax.set_ylim(55, 90)
    ax.set_xlim(0, 4 * 4)
    return fig, ax

def update_live_plot(ax, current_time, pitch, duration):
    if pitch is not None:
        ax.hlines(pitch, current_time, current_time + duration, colors='black', linewidth=4)
        plt.draw()
        plt.pause(duration * 0.4)

# music21 sheet
def show_final_sheet(musical_score, bpm=120):
    s = stream.Stream()
    s.append(tempo.MetronomeMark(number=bpm))
    s.append(meter.TimeSignature('4/4'))

    # Force the key signature to C major
    s.append(m21key.Key('C'))

    for bar in musical_score:
        for (pitch_val, duration, volume) in bar:
            if pitch_val is None:
                n = note.Rest(quarterLength=duration)
            else:
                n = note.Note(quarterLength=duration)
                n.pitch = m21pitch.Pitch()
                n.pitch.midi = int(pitch_val)
                if n.pitch.accidental is not None:
                    n.pitch.accidental = None
            s.append(n)

    def open_musescore():
        s.show()  # s.show freezes the python demo window, so thread it to avoid

    # program will stay alive while musescore is open
    t = threading.Thread(target=open_musescore)
    t.start()


env = RLMusicBotEnv()
obs, _ = env.reset()

fig, ax = init_live_plot()
current_time = 0.0

done = False

while not done:
    remaining = 4 - sum([note[1] for note in env._musical_score[env.current_bar]])
    if remaining <= 0:
        env.current_bar += 1
        if env.current_bar >= env.bars:
            break
        continue

    action = env.action_space.sample()
    new_note = env._map_action_to_note(action)
    duration = new_note[1]

    if duration > remaining:
        continue

    obs, reward, done, _, _ = env.step(action)

    pitch = new_note[0]
    update_live_plot(ax, current_time, pitch, duration)
    current_time += duration

print(env._musical_score)
print(reward)

# Save to MIDI
env.save_to_midi("random_song.mid")
show_final_sheet(env._musical_score)

# Keep the Matplotlib window alive properly, also needed so the python window doesnt freeze due to s.show()
plt.ioff()
plt.show()