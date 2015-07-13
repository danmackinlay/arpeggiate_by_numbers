from config import *
import pandas as pd
import numpy as np
import pickle

def get_note_event_store():
    return pd.HDFStore(NOTE_EVENT_TABLE_PATH, 'r')

def get_note_events(where=None):
    with pd.HDFStore(NOTE_EVENT_TABLE_PATH, 'r') as st:
        return st.select('note_event_meta', where=None)

def get_one_song(song_name=None):
    with pd.HDFStore(NOTE_EVENT_TABLE_PATH, 'r') as st:
        if song_name is None:
            song_name = str(st.select("note_event_meta", stop=1).song.iloc[0])
        frame = pd.concat([ #ignore internal index
            st.select('note_event_meta',
                where="song=song_name",
                columns=["pitch", "time"])
        ], ignore_index=True)
        return frame

def iter_evented_songs():
    with pd.HDFStore(NOTE_EVENT_TABLE_PATH, 'r') as st:
        song_names = st.select("note_event_meta", columns=["song"])["song"].unique()
        for song_name in song_names:
            frame = pd.concat([ #ignore internal index
                st.select('note_event_meta',
                where="song=song_name",
                columns=["pitch", "time"])
            ], ignore_index=True)
        yield song_name, frame

def encode_notes(verbose=False, mode="w"):
    from music21 import converter, instrument, midi
    from music21.note import Note, NotRest, Rest
    from music21.chord import Chord
    from random import randint, sample
    with pd.HDFStore(
        NOTE_EVENT_TABLE_PATH,
        complevel=5,
        complib="blosc",
        chunksize=2<<18, #512kb is recommended for large blosc tables
        mode=mode,
    ) as note_event_store:
        def parse_midi_file(base_dir, midi_file):
            """workhorse function"""
            midi_in_file = os.path.join(base_dir, midi_file)
            file_key = midi_in_file[len(MIDI_BASE_DIR):-4] #only the basename
            if verbose:
                print "parsing", base_dir, file_key
            note_stream = converter.parse(midi_in_file)
            all_times = []
            all_pitches = []
            for next_item in note_stream.flat.notes.offsetMap:
                event = next_item.element
                #only handle Notes and Chords
                if not isinstance(event, NotRest):
                    continue
                next_time_stamp = next_item.offset

                if hasattr(event, 'pitch'):
                    pitches = [event.pitch.midi]
                if hasattr(event, 'pitches'):
                    pitches = sorted([p.midi for p in event.pitches])
                    ## OR: randomize order:
                    #pitches = random.sample(pitches, len(pitches))
                for pitch in pitches:
                    all_times.append(next_time_stamp)
                    all_pitches.append(pitch)
            frame = pd.DataFrame(dict(
                time=np.asarray(all_times, dtype="float32"),
                pitch=np.asarray(all_pitches, dtype="int32"),
            ))
            frame['song'] = file_key
        
            note_event_store.append('note_event_meta',
                frame,
                expectedrows=50000,
                min_itemsize={'song':50},
                data_columns=["song", "time"],
                index=False, #suppress index until end
            )
            

        def parse_if_midi(_, file_dir, file_list):
            for f in file_list:
                if f.lower().endswith('mid'):
                    parse_midi_file(file_dir, f)

        os.path.walk(MIDI_BASE_DIR, parse_if_midi, None)
        if verbose:
            print "indexing"

        note_event_store.create_table_index('note_event_meta',
            columns=["song", "time"],
            optlevel=9, kind='full')
    return get_note_events()


def saveobj(obj, filename):
    from config import DATA_DIR_TMP
    with open(os.path.join(
        DATA_DIR_TMP, "{}.pkl".format(filename)), 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def loadobj(filename):
    from config import DATA_DIR_TMP
    with open(os.path.join(
        DATA_DIR_TMP, "{}.pkl".format(filename)), 'rb') as input:
        obj = pickle.load(input)
    return obj
