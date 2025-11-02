from RLMusicEnv import RLMusicBotEnv
from final_visualization import show_final_sheet

env = RLMusicBotEnv(render_mode="human")

for _ in range(2):
    obs, _ = env.reset()
    done = False
    while not done:

        # TODO refact beats left calculation as env observation
        #Calculate how many beats left in bar
        remaining = 4 - sum([note[1] for note in env._musical_score[env.current_bar]])
        if env.current_bar >= env.bars:
            break

        #Pick action
        action = env.action_space.sample()
        new_note = env._map_action_to_note(action)
        duration = new_note[1]

        #Filter out notes that don't fit
        if duration > remaining:
            continue

        #Step
        obs, reward, done, _, _ = env.step(action)

    print("musical score: ")
    print(env._musical_score)
    print("chord progression: ")
    print(", ".join(env.chord_progression))
    print("reward: ")
    print(reward)

# Save to MIDI
env.save_to_midi("random_song.mid")
# show_final_sheet(env._musical_score)