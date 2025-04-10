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


type_map = {
    "ANTI_SACCADE" : {
        'experiment' : "string",
        'participant_id' : "int64", 
        'trial_id' : "int64", 
        'time' : "int64", 
        'event' : "string",
        'eye' : "string",
        'colour' : "string",
        'stimulus_x' : "float64", 
        'stimulus_y' : "float64", 
        'start_time' : "float64", 
        'end_time' : "float64", 
        'duration' : "float64", 
        'x' : "float64", 
        'y' : "float64", 
        'avg_pupil_size' : "float64", 
        'start_x' : "float64", 
        'start_y' : "float64", 
        'end_x' : "float64", 
        'end_y' : "float64", 
        'amplitude' : "float64", 
        'peak_velocity' : "float64", 
        'side' : "string",
        'time_elapsed' : "float64", 
        'delay' : "float64"},
    "EVIL_BASTARD" : {
        'experiment' :"string", 
        'participant_id' : "int64", 
        'trial_id' : "int64", 
        'time' : "int64", 
        'event' : "string", 
        'eye' : "string",
        'start_time' : "float64", 
        'end_time' : "float64", 
        'duration' : "float64", 
        'x' : "float64", 
        'y' : "float64", 
        'avg_pupil_size' : "float64",
        'start_x' : "float64", 
        'start_y' : "float64", 
        'end_x' : "float64", 
        'end_y' : "float64", 
        'amplitude' :  "float64", 
        'peak_velocity' : "float64",
        'angle' : "float64", 
        'speed' : "float64", 
        'target_x' : "float64", 
        'target_y' : "float64", 
        'colour' : "string", 
        'stimulus_x' : "float64",
        'stimulus_y' : "float64"
    }
}

type_map_samples = {
    "experiment" : "string",
    "participant_id" : "int64",
    "trial_id" : "int64",
    "time" : "int64",
    "x_left" : "float64",
    "y_left" : "float64",
    "pupil_size_left" : "float64",
    "x_velocity_left" : "float64",
    "y_velocity_left" : "float64",
    "x_right" : "float64",
    "y_right" : "float64",
    "pupil_size_right" : "float64",
    "x_velocity_right" : "float64",
    "y_velocity_right" : "float64",
    "x_resolution" : "float64",
    "y_resolution" : "float64",
    "error_message" : "string"
    }