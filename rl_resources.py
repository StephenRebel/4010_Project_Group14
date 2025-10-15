import pretty_midi
from music21 import pitch

# midi to str of midivalues (81,71,etc)
def inspect_midi(midi_path:str) -> list[str]: 
    nstr = []
    pm = pretty_midi.PrettyMIDI(midi_path)
    for inst in pm.instruments:
        for n in inst.notes:
            nstr.append(n.pitch)
    return nstr

# midi to str of just notes (A4, C5, etc) purely for scale and melody purposes
# all notes will be natural until a key is defined (I think)
def midi_to_notes(midi_path: str) -> list[str]:
    pm = pretty_midi.PrettyMIDI(midi_path)
    note_names = []
    for instrument in pm.instruments:
        for n in instrument.notes:
            m21_pitch = pitch.Pitch(midi=n.pitch)
            note_names.append(m21_pitch.nameWithOctave)
    return note_names

# tester func
def main():
    input_midi = "random_song.mid"
    inspect = inspect_midi(input_midi)
    print(inspect)
    notes = midi_to_notes(input_midi)
    print(notes)

if __name__ == "__main__":
    main()
