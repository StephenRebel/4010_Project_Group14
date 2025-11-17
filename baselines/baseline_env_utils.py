EOS_ID = 320 # Hardcoded env.action_space.n + 1

# Analogus to _map_action_to_note and the reverse form the environment
def action_from_note_id(pitch_id, duration_id, volume_id, env):
    mult = env.n_durations * env.n_volumes
    return int(pitch_id * mult + duration_id * env.n_volumes + volume_id)

def note_id_from_action(action, env):
    mult = env.n_durations * env.n_volumes
    pitch_id = action // mult
    duration_id = (action % mult) // env.n_volumes
    volume_id = action % env.n_volumes

    return pitch_id, duration_id, volume_id

# Converting from musical score representation to list of actions helpers
def musical_score_to_actions(musical_score, env):
    actions = []

    for bar in musical_score:
        for (pitch, duration, volume) in bar:
            pitch_id = env.rest_action if pitch is None else env.pitches.index(pitch)
            duration_id = env.durations.index(duration)
            volume_id = env.volumes.index(volume)

            actions.append(action_from_note_id(pitch_id, duration_id, volume_id, env))

    return actions

def actions_to_musical_score(actions, env, eos_token=EOS_ID):
    musical_score = [[] for _ in range(env.bars)]
    current_bar = 0
    current_beats = 0.0

    for a in actions:
        if eos_token is not None and a == eos_token:
            break

        pitch_id, duration_id, volume_id = note_id_from_action(a, env)
        pitch = None if pitch_id == env.rest_action else env.pitches[pitch_id]
        duration = env.durations[duration_id]
        volume = env.volumes[volume_id]

        # Handle musical structuring
        if current_bar >= env.bars:
            break

        if current_beats + duration > env.beats_per_bar:
            current_bar += 1
            current_beats = 0.0

            if current_bar >= env.bars:
                break

        musical_score[current_bar].append((pitch, duration, volume))
        current_beats += duration
    
    return musical_score

def model_vocab_mapping(dataset, env):
    eos_id = EOS_ID
    mapping = [sequence + [eos_id] for sequence in dataset]
    vocab_size = eos_id + 1

    return mapping, eos_id, vocab_size