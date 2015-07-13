import os.path
import random

# keep everything reproducible
random.seed(12345)

#when calculating onset statistics, aggregate notes this close together
ONSET_TOLERANCE = 0.06

MIDI_BASE_DIR = os.path.expanduser('~/Music/midi/rag/')
OUTPUT_BASE_PATH = os.path.normpath("./")
NOTE_EVENT_TABLE_PATH = os.path.join(OUTPUT_BASE_PATH, 'rag_events.h5')
NOTE_OBS_TABLE_PATH = os.path.join(OUTPUT_BASE_PATH, 'rag_obs.h5')
DATA_DIR_TMP = os.path.expanduser('~/Dropbox/swap/')
