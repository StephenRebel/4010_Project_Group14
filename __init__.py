import numpy as np
import gymnasium as gym
import pretty_midi
import matplotlib.pyplot as plt
from music21 import stream, note, meter, tempo
import threading


class RLMusicBotEnv(gym.Env):
    def __init__(self, bars: int = 4):
        super(RLMusicBotEnv, self).__init__()

        # Config
        self.bars = bars
        self.beats_per_bar = 4
        self.scale = "C"

        # Note options
        self.durations = [ 0.25, 0.5, 1.0, 2.0, 4.0]
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
        obs = np.zeros((1,), dtype=np.float32)
        return obs, {}
    
    def _get_obs(self):
        return np.array([self._musical_score])

    def step(self, action):
        note = self._map_action_to_note(action)
        self._musical_score[self.current_bar].append(note)

        total_duration = sum(n[1] for n in self._musical_score[self.current_bar] if n[0] is not None)
        if total_duration >= self.beats_per_bar:
            self.current_bar += 1

        done = self.current_bar >= self.bars
        reward = 0.0
        obs = np.zeros((1,), dtype=np.float32)
        info = {}
        return obs, reward, done, False, info
    
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

    for bar in musical_score:
        for (pitch, duration, volume) in bar:
            if pitch is None:
                n = note.Rest(quarterLength=duration)
            else:
                n = note.Note(pitch, quarterLength=duration)
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

# Save to MIDI
env.save_to_midi("random_song.mid")
show_final_sheet(env._musical_score)

# Keep the Matplotlib window alive properly, also needed so the python window doesnt freeze due to s.show()
plt.ioff()
plt.show()