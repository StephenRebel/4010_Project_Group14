import os
import json
import pretty_midi
import bisect
import numpy as np
from tqdm import tqdm

from RLMusicEnv import RLMusicBotEnv
from baselines.baseline_env_utils import action_from_note_id

# Helpful for many of the pretty midi commands: https://nbviewer.org/github/craffel/pretty-midi/blob/main/Tutorial.ipynb
# nottingham-dataset: https://github.com/jukedeck/nottingham-dataset/tree/master/MIDI/melody

# Filepaths and configs
MIDI_FOLDER = "baselines/nottingham_midi/"
OUT_FILE = "baselines/baseline_dataset_cleaned.jsonl"
MIN_NOTES_PER_FILE = 16
VOLUME_ID = 2 
EPS = 1e-6
BAD_DURATION_THRESHOLD = 0.15 
MAX_FORCE_RATIO = 0.10

env = RLMusicBotEnv(bars=8) 
ENV_PITCHES = env.pitches
ENV_DURATIONS = env.durations
ENV_N_VOLUMES = env.n_volumes

def is_compatible_time_signature(pm):
    # Check if this time signature is useable, we work in 4/4
    # so anything x/4 at all points in song is acceptable

    # Assume 4/4
    if not pm.time_signature_changes:
        return True
        
    # Check every time signature change to ensure x/4
    for ts in pm.time_signature_changes:
        if ts.denominator != 4:
            return False
        
    return True

def is_monophonic_melody(pm):
    # If melody is not monophonic drop it
    notes = []

    for inst in pm.instruments:
        notes.extend(inst.notes)

    notes = sorted(notes, key=lambda n: n.start)

    prev_note_end = -EPS
    for note in notes:
        if note.start < prev_note_end - EPS:
            return False

        prev_note_end = max(note.end, prev_note_end)

    return True

def best_transpose_for_pitchset(pitches, target_set):
    # Identify the amount to shift notes by to have them in range of our desired scale
    best_shift = 0
    best_count = -1

    for shift in range(-12, 13):
        shifted = [pitch + shift for pitch in pitches]
        in_range_pitches = sum(1 for pitch in shifted if pitch in target_set)

        if in_range_pitches > best_count:
            best_count = in_range_pitches
            best_shift = shift

    return best_shift

def octave_shift_into_set(pitch, target_set):
    # Identify if a shift of the octave needed for a pitch to be in target range
    # return closest pitch to ensure valid action if nothing else
    for octave in range(-3, 4):
        cand = pitch + 12 * octave

        if cand in target_set:
            return cand
        
    return min(target_set, key=lambda x: abs(x - pitch))

def get_note_duration_in_beats(start_time, end_time, tempo_change_times, tempo_bpms):
    # Find the tempo segment this note is played in
    index = bisect.bisect_right(tempo_change_times, start_time) - 1
    
    current_tempo = tempo_bpms[index]
    
    # Calculate number of beats note takes given its duration and current song tempo
    duration_sec = end_time - start_time
    duration_beats = duration_sec * (current_tempo / 60.0)
    
    return duration_beats

def split_into_allowed(computed_duration):
    # Split a note into parts based on allowed environment durations
    note_durations = []
    remaining_duration = computed_duration
    valid_durations_sorted = sorted(ENV_DURATIONS, reverse=True)

    # Try to split into biggest sections first
    while remaining_duration > EPS:
        for duration in valid_durations_sorted:
            if remaining_duration + EPS >= duration:
                note_durations.append(duration)
                remaining_duration -= duration
                break

    return note_durations

def get_duration_or_decompose(computed_duration):
    # Set the note to an appropriate valid duration, or
    # Decompose the note if does not fit with environment durations
    # Return list of durations and whether it was forced or not

    # Try to find good note durations
    nearest_duration = min(ENV_DURATIONS, key=lambda x: abs(x - computed_duration))
    duration_error = abs(nearest_duration - computed_duration)
    
    if duration_error < BAD_DURATION_THRESHOLD:
        return [nearest_duration], False

    # Try to decompose note
    split_parts = split_into_allowed(computed_duration)
    split_sum = sum(split_parts)
    split_error = abs(split_sum - computed_duration)

    # If the split notes have less error return them
    if split_error < duration_error and split_error < BAD_DURATION_THRESHOLD:
        return split_parts, False

    # Fallback to clest durations
    return [nearest_duration], True

def process_midi_to_sequence(path):
    # Process a midi file ensure valid time signature and monophonic
    # Parse into action format after correcting for pitch and octives as needed
    try:
        pm = pretty_midi.PrettyMIDI(path)
    except Exception:
        return None

    # Filter time signatures (x/4 types)
    if not is_compatible_time_signature(pm):
        return None

    # Filter for monophonic melody songs
    if not is_monophonic_melody(pm):
        return None

    melody = pm.instruments[0].notes

    # Determine best transpositions for notes to fit in our environments pitch range
    raw_pitches = [note.pitch for note in melody]
    shift = best_transpose_for_pitchset(raw_pitches, set(ENV_PITCHES))

    # Get changes in tempo or default to 120 BPM
    times, tempo_changes = pm.get_tempo_changes()
    if len(times) == 0:
        times = np.array([0.0])
        tempo_changes = np.array([120.0])

    # Composition actions
    action_sequence = []

    # Counters to ensure we are accurately transcribing rhythm
    total_notes_processed = 0
    forced_durations = 0

    for note in melody:
        # Ensure pitches valid
        pitch_shifted = note.pitch + shift
        if pitch_shifted not in ENV_PITCHES:
            pitch_shifted = octave_shift_into_set(pitch_shifted, ENV_PITCHES)
        
        # Account for duration of note based on tempo and current time
        computed_duration = get_note_duration_in_beats(note.start, note.end, times, tempo_changes)
        total_notes_processed += 1
        
        # Get the nearest valid note duration(s) for computed duration
        note_durations, is_forced = get_duration_or_decompose(computed_duration)

        if is_forced:
            forced_durations += 1

        # Generate action integers for note(s)
        pitch_id = ENV_PITCHES.index(pitch_shifted)

        for duration in note_durations:
            duration_id = ENV_DURATIONS.index(duration)
            action = action_from_note_id(pitch_id, duration_id, VOLUME_ID, env)
            action_sequence.append(action)

    # Drop any songs where too much was forced
    if total_notes_processed > 0:
        forced_ratio = forced_durations / total_notes_processed
        if forced_ratio > MAX_FORCE_RATIO:
            return None

    return action_sequence

def run(): 
    valid_count = 0
    skipped_count = 0
    
    # Loop over all midi files, transform and keep sequences of valid ones
    with open(OUT_FILE, "w") as f:
        for filename in tqdm(os.listdir(MIDI_FOLDER)):
            path = os.path.join(MIDI_FOLDER, filename)
            sequence = process_midi_to_sequence(path)
            
            if sequence:
                composition = {"id": os.path.basename(path), "actions": sequence}
                f.write(json.dumps(composition) + "\n")
                valid_count += 1
            else:
                skipped_count += 1

    print(f"Processing Complete.")
    print(f"Kept {valid_count} files, dropped {skipped_count} files.")

if __name__ == "__main__":
    run()