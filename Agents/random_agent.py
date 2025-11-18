import os

class RandomAgent:
    def __init__(self, env):
        self.env = env

    def test(self, render=False, midi_filename=None, debug=False):
        self.env.reset()
        done = False
        while not done:
            action = self.env.action_space.sample()
            _, _, done, _, _ = self.env.step(action)
            if render and self.env.render_mode == "human":
                self.env.render()
                
        final_reward = self.env._compute_reward(self.env._musical_score, debug=debug)

        print("\nRandom test completed.")
        print("musical score: ")
        print(self.env._musical_score)
        print("chord progression: ")
        print(", ".join(self.env.chord_progression))
        print("reward: ")
        print(final_reward)

        if midi_filename:
            os.makedirs(os.path.dirname(midi_filename), exist_ok=True)
            self.env.save_to_midi(midi_filename)

