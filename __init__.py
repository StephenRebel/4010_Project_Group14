import numpy as np
import gymnasium as gym

class RLMusicBotEnv(gym.Env):

    # Do we specify the scale, and time signature? or is that inherent to us.
    def __init__(self, bars: int = 4):
        # Number of bars to generate
        self.bars = bars

        # 4/4 time signature
        self.beats_per_bar = 4 # 4 beats per bar
        self.note_value = 4 # Quarter note gets the beat

        self.scale = "C"

        # Initialize the musical score as state
        self._musical_score = []

        # Define what agent can observe
        self.observation_space = gym.spaces.Dict({
            # Current musical score
            # Maybe others
        })

        # Define what actions are available
        self.durations = [1.0, 2.0, 3.0, 4.0] # quarter, half, dotted half, whole
        self.n_durations = len(self.durations)

        self.volumes = [0.2, 0.4, 0.6, 0.8, 1.0] # 5 levels of volume, velocity in MIDI, Keep for now could remove as not very relevant
        self.n_volumes = len(self.volumes)

        self.pitches = list(range(0, 128)) # MIDI pitches from 0 to 127. NOTE may be subject to change, maybe limit to a scale certain range
        self.rest_action = len(self.pitches) # An action to represent a rest
        self.n_pitches = len(self.pitches) + 1 # +1 for rest action

        self.action_space = gym.spaces.MultiDiscrete([
            self.n_pitches,
            self.n_durations,
            self.n_volumes
        ])

        # Map action values to notes
        def _map_action_to_note(self, action):
            pitch_idx, duration_idx, volume_idx = action

            if pitch_idx == self.rest_action:
                pitch = None
            else:
                pitch = self.pitches[pitch_idx]
            duration = self.durations[duration_idx]
            volume = self.volumes[volume_idx]

            return (pitch, duration, volume)