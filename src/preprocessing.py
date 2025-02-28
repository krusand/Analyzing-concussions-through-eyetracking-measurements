# Libraries
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import *

def get_events_from_trial_id_line(participant_id,line,experiment):
    linesplit = line.split()
    current_trial_id = int(linesplit[-1])
    time = int(linesplit[1])
    return {"experiment": experiment
                    ,"participant_id": participant_id
                    , "trial_id": current_trial_id
                    , "time": time
                    , "event": "TRIALID"
                    , "colour": None
                    , "coordinates": None}
    
def get_events_from_start_line(participant_id, current_trial_id, line,experiment):
    linesplit = line.split()
    time = int(linesplit[1])
    return {"experiment": experiment
                    ,"participant_id": participant_id
                    , "trial_id": current_trial_id
                    , "time": time
                    , "event": "START"
                    , "colour": None
                    , "coordinates": None}

def get_events_from_end_line(participant_id, current_trial_id, line,experiment):
    linesplit = line.split()
    time = int(linesplit[1])
    return {"experiment": experiment
                    ,"participant_id": participant_id
                    , "trial_id": current_trial_id
                    , "time": time
                    , "event": "END"
                    , "colour": None
                    , "coordinates": None}

def get_events_from_trial_var_data_line(participant_id, current_trial_id, line,experiment):
    linesplit = line.split()
    time = int(linesplit[1])
    stimulus_onset_in_seconds = float(line.split()[-1])
    stimulus_onset_in_milliseconds = int(stimulus_onset_in_seconds * 1000)
    coordinates = " ".join(linesplit[4:5])
    return {"experiment": experiment
                    ,"participant_id": participant_id
                    , "trial_id": current_trial_id
                    , "time": time
                    , "event": "TRIAL_VAR_DATA"
                    , "colour": None
                    , "coordinates": coordinates
                    , "stimulus_onset_ms": stimulus_onset_in_milliseconds}


def get_events_from_fixpoint_line(participant_id, current_trial_id, line, experiment):
    linesplit = line.split()
    time = int(linesplit[1])
    colour = " ".join(linesplit[4:7])
    coordinates = " ".join(linesplit[10:12])
    return {"experiment": experiment
                    ,"participant_id": participant_id
                    , "trial_id": current_trial_id
                    , "time": time
                    , "event": "FIXPOINT"
                    , "colour": colour
                    , "coordinates": coordinates}

def process_line(line, participant_id, current_trial_id, experiment):
    # Detect trial start
    if "TRIALID" in line:
        event = get_events_from_trial_id_line(participant_id, line, experiment)
        current_trial_id = event["trial_id"]
        return [event, current_trial_id]
    # Detect beginning of Trial
    elif line.startswith("START"):
        event = get_events_from_start_line(participant_id, current_trial_id, line, experiment)
        return [event]
    elif line.startswith("END"):
        event = get_events_from_end_line(participant_id, current_trial_id, line, experiment)
        return [event]
    elif "!V TRIAL_VAR_DATA" in line:
        event = get_events_from_trial_var_data_line(participant_id, current_trial_id, line, experiment)
        return [event]
    elif "FIXPOINT" in line:
        event = get_events_from_fixpoint_line(participant_id, current_trial_id, line, experiment)
        return [event]
    else:
        return [None]

def add_stimulus_events(events:pd.DataFrame, participant_id, experiment):
    stimulus_onset_events = []
    red_fixpoints = events[(events['event'] == "FIXPOINT") & (events["colour"] == RED)]
    white_fixpoints = events[(events['event'] == "FIXPOINT") & (events["colour"] == WHITE)]

    for _, row in red_fixpoints.iterrows():
        current_trial_id = row["trial_id"]

        white_fixpoints_current_trial = white_fixpoints[white_fixpoints["trial_id"] == current_trial_id]

        if not white_fixpoints_current_trial.empty:
            fixpoint_white_time = int(white_fixpoints_current_trial.iloc[0]["time"])
            stimulus_onset = events[(events["trial_id"] == current_trial_id) & (events["event"].str.startswith("TRIAL_VAR_DATA"))]
            if not stimulus_onset.empty:
                stimulus_onset_ms = int(stimulus_onset.iloc[0]["stimulus_onset_ms"])
                stimulus_onset_time = fixpoint_white_time + stimulus_onset_ms
        
        stimulus_onset_events.append({
                "experiment": experiment
                , "participant_id": participant_id
                , "trial_id": current_trial_id
                , "time": stimulus_onset_time
                , "event": "STIMULUS_ONSET"
                , "colour": row["colour"]
                , "coordinates": row["coordinates"]
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

    for line in lines:
        line = line.strip()
        line_event = process_line(line=line,
                     participant_id=participant_id,
                     current_trial_id=current_trial_id,
                     experiment=experiment)
        if line_event[0] is None:
            continue
        elif len(line_event) > 1:
            current_trial_id = line_event[1]
        events.append(line_event[0])

    events = pd.DataFrame(events)
    
    print("Finished initial event loading")
    print("Adding stimulus events")

    events = add_stimulus_events(events, participant_id, experiment)
    return events.sort_values(by = ["participant_id", "trial_id", "time"]).reset_index(drop=True)



def main():
    #participant_metadata = pd.read_excel(DATA_DIR / "demographic_info.xlsx")
    asc_files = [f for f in os.listdir(ASC_RAW_DIR) if f.endswith('.asc') and "anti" in f]
    print(process_asc_file(asc_files[-1], experiment="ANTI_SACCADE"))




if __name__ == '__main__':
    main()

