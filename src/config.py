from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
ASC_DIR = DATA_DIR / "asc_files"
ASC_RAW_EVENTS_DIR = ASC_DIR / "raw_events"
ASC_RAW_SAMPLES_DIR = ASC_DIR / "raw_samples"
METADATA_DIR = DATA_DIR / "metadata"
METADATA_RAW_DIR = METADATA_DIR / "raw"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
FEATURES_DIR = DATA_DIR / "features"

LOG_DIR = PROJ_ROOT / "log"

RED = "255 0 0"
GREEN = "0 255 0"
BLUE = "0 0 255"
WHITE = "255 255 255"


# Prperocessing
numeric_columns_anti_saccade = ['participant_id', 'trial_id', 'time', 'stimulus_x', 'stimulus_y', 'start_time', 'end_time', 
                                'duration', 'x', 'y', 'avg_pupil_size', 'start_x', 'start_y', 'end_x', 'end_y', 'amplitude', 
                                'peak_velocity', 'time_elapsed', 'delay']