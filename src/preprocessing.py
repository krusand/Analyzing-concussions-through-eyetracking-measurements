# Libraries
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import *


# Fittslaw          : MSG	2967684 TRIAL_VAR_LABELS distance target_width
# Anti-saccade      : MSG	1913870 TRIAL_VAR_LABELS side delay
# Reactions         : MSG   1189006 TRIAL_VAR_LABELS pos_x pos_y delay
# Fixations         : MSG	1443219 TRIAL_VAR_LABELS pos_x pos_y target_shape
# King-devick       : MSG	2661907 TRIAL_VAR_LABELS marks time_elapsed
# Patterns          : MSG	3983900 TRIAL_VAR_LABELS angle speed
# Shapes            : MSG	1322015 TRIAL_VAR_LABELS shape
# Smooth Pursuits   : MSG	3768702 TRIAL_VAR_LABELS shape speed


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

def add_stimulus_events(events:pd.DataFrame, participant_id, experiment):
    stimulus_onset_events = []
    red_fixpoints = events[(events['event'] == "FIXPOINT") & (events["colour"] == RED) & (events["experiment"] == "ANTI-SACCADE")]
    white_fixpoints = events[(events['event'] == "FIXPOINT") & (events["colour"] == WHITE) & (events["experiment"] == "ANTI-SACCADE")]

    for _, row in red_fixpoints.iterrows():
        current_trial_id = row["trial_id"]

        white_fixpoints_current_trial = white_fixpoints[white_fixpoints["trial_id"] == current_trial_id]

        if not white_fixpoints_current_trial.empty:
            fixpoint_white_time = int(white_fixpoints_current_trial.iloc[0]["time"])
            stimulus_onset = events[(events["trial_id"] == current_trial_id) & (events["event"].str.startswith("TRIAL_VAR_DATA"))]
            if not stimulus_onset.empty:
                if "time_elapsed" in stimulus_onset.columns: 
                    stimulus_onset_ms = int(float(stimulus_onset.iloc[0]["time_elapsed"]) * 1000)
                elif "delay" in stimulus_onset.columns:
                    stimulus_onset_ms = int(float(stimulus_onset.iloc[0]["delay"]) * 1000)
                stimulus_onset_time = fixpoint_white_time + stimulus_onset_ms
        
        stimulus_onset_events.append({
                "experiment": experiment
                , "participant_id": participant_id
                , "trial_id": current_trial_id
                , "time": stimulus_onset_time
                , "event": "STIMULUS_ONSET"
                , "colour": row["colour"]
                , "stimulus_x": row["stimulus_x"]
                , "stimulus_y": row["stimulus_y"]
        })
    
    events = pd.concat([events, pd.DataFrame(stimulus_onset_events)], ignore_index=True)
    return events

def process_asc_file(filename, experiment):
    print(f"Processing {filename}")

    filepath = ASC_RAW_DIR / filename
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

    if experiment == "ANTI-SACCADE":
        print("Adding stimulus events")
        events = add_stimulus_events(events, participant_id, experiment)
    return events.sort_values(by = ["participant_id", "trial_id", "time"]).reset_index(drop=True)

def process_asc_files(files, experiment):
    event_dfs = []
    for file in tqdm(files):
        event_dfs.append(process_asc_file(filename=file, experiment=experiment))
    return pd.concat(event_dfs)


def main():
    file_filters = ["anti-saccade", "FittsLaw", "Fixations", "KingDevick", "Patterns", "Reaction", "Shapes", "SmoothPursuits"]
    experiments = ["ANTI_SACCADE" , "FITTS_LAW", "FIXATIONS", "KING_DEVICK", "EVIL_BASTARD", "REACTION", "SHAPES", "SMOOTH_PURSUITS"]
    #participant_metadata = pd.read_excel(DATA_DIR / "demographic_info.xlsx")
    for file_filter, experiment in zip(file_filters, experiments):
        asc_files = [f for f in os.listdir(ASC_RAW_DIR) if f.endswith('.asc') and f.startswith(f"{file_filter}")]
        df = process_asc_files(asc_files, experiment=experiment)
        path_save = PROCESSED_DIR / f"{experiment}.pq"
        print(f"Saving to {path_save}")
        df.to_parquet(path_save, index=False)

if __name__ == '__main__':
    main()

