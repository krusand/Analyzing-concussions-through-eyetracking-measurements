# Libraries
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from config import *

def get_events_from_trial_var_labels_line(line):
    linesplit= line.split()
    return linesplit[3:]

def get_events_from_trial_id_line(participant_id,line,experiment):
    linesplit = line.split()
    current_trial_id = int(linesplit[-1])
    time = int(linesplit[1])

    return {"experiment": experiment
            ,"participant_id": participant_id
            , "trial_id": current_trial_id
            , "time": time
            , "event": "TRIALID"}
 
def get_events_from_start_line(participant_id, current_trial_id, line,experiment):
    linesplit = line.split()
    time = int(linesplit[1])
    
    return {"experiment": experiment
                    ,"participant_id": participant_id
                    , "trial_id": current_trial_id
                    , "time": time
                    , "event": "START"}

def get_events_from_end_line(participant_id, current_trial_id, line,experiment):
    linesplit = line.split()
    time = int(linesplit[1])
    return {"experiment": experiment
                    ,"participant_id": participant_id
                    , "trial_id": current_trial_id
                    , "time": time
                    , "event": "END"}

def get_events_from_trial_var_data_line(participant_id, current_trial_id, line,experiment, trial_var_labels):
    linesplit = line.split()
    time = int(linesplit[1])
    trial_line_events = linesplit[-len(trial_var_labels):]
    trial_line_events_dict = {var_label: trial_event_value for var_label, trial_event_value in zip(trial_var_labels, trial_line_events)}

    return {"experiment": experiment
                    ,"participant_id": participant_id
                    , "trial_id": current_trial_id
                    , "time": time
                    , "event": "TRIAL_VAR_DATA"
                    , **trial_line_events_dict}

def get_events_from_fixpoint_line(participant_id, current_trial_id, line, experiment):
    linesplit = line.split()
    time = int(linesplit[1])
    colour = " ".join(linesplit[4:7])
    stimulus_x = float(linesplit[10])
    stimulus_y = float(linesplit[11])
    return {"experiment": experiment
                    ,"participant_id": participant_id
                    , "trial_id": current_trial_id
                    , "time": time
                    , "event": "FIXPOINT"
                    , "colour": colour
                    , "stimulus_x": stimulus_x
                    , "stimulus_y": stimulus_y}

def get_events_from_efix_line(participant_id, current_trial_id, line, experiment):
    linesplit = line.split()
    return {
                "experiment": experiment,
                "participant_id": participant_id,
                "trial_id": current_trial_id,
                "event": "EFIX",
                "eye": linesplit[1] if len(linesplit) > 1 else np.nan,
                "start_time": int(linesplit[2]) if len(linesplit) > 2 and linesplit[2].isdigit() else np.nan,
                "end_time": int(linesplit[3]) if len(linesplit) > 3 and linesplit[3].isdigit() else np.nan,
                "duration": int(linesplit[4]) if len(linesplit) > 4 and linesplit[4].isdigit() else np.nan,
                "x": float(linesplit[5]) if len(linesplit) > 5 and linesplit[5].replace('.', '', 1).isdigit() else np.nan,
                "y": float(linesplit[6]) if len(linesplit) > 6 and linesplit[6].replace('.', '', 1).isdigit() else np.nan,
                "avg_pupil_size": float(linesplit[7]) if len(linesplit) > 7 and linesplit[7].replace('.', '', 1).isdigit() else np.nan,
            }

def get_events_from_sfix_line(participant_id, current_trial_id, line, experiment):
    linesplit = line.split()
    time = int(linesplit[-1])
    return {"experiment": experiment
            , "participant_id": participant_id
            , "trial_id": current_trial_id
            , "time": time
            , "event": "SFIX"
            , "eye": linesplit[1]}

def get_events_from_esacc_line(participant_id, current_trial_id, line, experiment):
    linesplit = line.split()
    return {
                "experiment": experiment,
                "participant_id": participant_id,
                "trial_id": current_trial_id,
                "event": "ESACC",
                "eye": linesplit[1] if len(linesplit) > 1 else np.nan,
                "start_time": int(linesplit[2]) if len(linesplit) > 2 and linesplit[2].isdigit() else np.nan,
                "end_time": int(linesplit[3]) if len(linesplit) > 3 and linesplit[3].isdigit() else np.nan,
                "duration": int(linesplit[4]) if len(linesplit) > 4 and linesplit[4].isdigit() else np.nan,
                "start_x": float(linesplit[5]) if len(linesplit) > 5 and linesplit[5].replace('.', '', 1).isdigit() else np.nan,
                "start_y": float(linesplit[6]) if len(linesplit) > 6 and linesplit[6].replace('.', '', 1).isdigit() else np.nan,
                "end_x": float(linesplit[7]) if len(linesplit) > 7 and linesplit[7].replace('.', '', 1).isdigit() else np.nan,
                "end_y": float(linesplit[8]) if len(linesplit) > 8 and linesplit[8].replace('.', '', 1).isdigit() else np.nan,
                "amplitude": float(linesplit[9]) if len(linesplit) > 9 and linesplit[9].replace('.', '', 1).isdigit() else np.nan,
                "peak_velocity": float(linesplit[10]) if len(linesplit) > 10 and linesplit[10].replace('.', '', 1).isdigit() else np.nan,
            }

def get_events_from_ssacc_line(participant_id, current_trial_id, line, experiment):
    linesplit = line.split()
    time = int(linesplit[-1])
    return {"experiment": experiment
            , "participant_id": participant_id
            , "trial_id": current_trial_id
            , "time": time
            , "event": "SSACC"
            , "eye": linesplit[1]}

def process_line(line, participant_id, current_trial_id, experiment, trial_var_labels):
    # Detect trial start
    if "TRIAL_VAR_LABELS" in line:
        event = get_events_from_trial_var_labels_line(line)
        return [event, "trial_var_labels"]
    elif "TRIALID" in line:
        event = get_events_from_trial_id_line(participant_id, line, experiment)
        current_trial_id = event["trial_id"]
        return [event, "trial_id", current_trial_id]
    # Detect beginning of Trial
    elif line.startswith("START"):
        event = get_events_from_start_line(participant_id, current_trial_id, line, experiment)
        return [event]
    elif line.startswith("END"):
        event = get_events_from_end_line(participant_id, current_trial_id, line, experiment)
        return [event]
    elif "!V TRIAL_VAR_DATA" in line:
        event = get_events_from_trial_var_data_line(participant_id, current_trial_id, line, experiment, trial_var_labels)
        return [event]
    elif "FIXPOINT" in line:
        event = get_events_from_fixpoint_line(participant_id, current_trial_id, line, experiment)
        return [event]
    elif line.startswith("EFIX"):
        event = get_events_from_efix_line(participant_id, current_trial_id, line, experiment)
        return [event]
    elif line.startswith("SFIX"):
        event = get_events_from_sfix_line(participant_id, current_trial_id, line, experiment)
        return [event]
    elif line.startswith("ESACC"):
        event = get_events_from_esacc_line(participant_id, current_trial_id, line, experiment)
        return [event]
    elif line.startswith("SSACC"):
        event = get_events_from_ssacc_line(participant_id, current_trial_id, line, experiment)
        return [event]
    else:
        return [None]


def process_asc_file(filename, experiment):
    print(f"Processing {filename}")

    filepath = ASC_RAW_EVENTS_DIR / filename
    participant_id = filename.split("_")[1]

    with open(filepath, 'r') as fp:
        lines = fp.readlines()

    events = []
    current_trial_id = None
    current_trial_var_labels = None

    for line in lines:
        line = line.strip()
        line_event = process_line(line=line,
                     participant_id=participant_id,
                     current_trial_id=current_trial_id,
                     experiment=experiment,
                     trial_var_labels=current_trial_var_labels)
        if line_event[0] is None: # no event in line
            continue
        elif len(line_event) > 1:
            if line_event[1] == "trial_var_labels":
                current_trial_var_labels = line_event[0]
                continue
            elif line_event[1] == "trial_id":
                current_trial_id = line_event[2]
        events.append(line_event[0])

    events = pd.DataFrame(events)
    
    print("Finished initial event loading")

    return events.sort_values(by = ["participant_id", "trial_id", "time"]).reset_index(drop=True)

def process_asc_files(files, experiment):
    event_dfs = []
    for file in tqdm(files):
        event_dfs.append(process_asc_file(filename=file, experiment=experiment))
    return pd.concat(event_dfs)

def run_asc_preprocessing():
    file_filters = ["anti-saccade", "FittsLaw", "Fixations", "KingDevick", "Patterns", "Reaction", "Shapes", "SmoothPursuits"]
    experiments = ["ANTI_SACCADE" , "FITTS_LAW", "FIXATIONS", "KING_DEVICK", "EVIL_BASTARD", "REACTION", "SHAPES", "SMOOTH_PURSUITS"]
    for file_filter, experiment in zip(file_filters, experiments):
        asc_files = [f for f in os.listdir(ASC_RAW_EVENTS_DIR) if f.endswith('.asc') and f.startswith(f"{file_filter}")]
        df = process_asc_files(asc_files, experiment=experiment)
        path_save = RAW_DIR / f"{experiment}.pq"
        print(f"Saving to {path_save}")
        df.to_parquet(path_save, index=False)

def main():
    # Convert asc files to parquet files
    run_asc_preprocessing()

if __name__ == '__main__':
    main()

