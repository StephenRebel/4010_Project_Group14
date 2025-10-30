from music21 import key as m21key, pitch as m21pitch
from music21 import stream, note, meter, tempo
import threading

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