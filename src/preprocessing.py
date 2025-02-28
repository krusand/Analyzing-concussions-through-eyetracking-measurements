# Libraries
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import *

def get_events_from_trial_id_line(participant_id,line):
    linesplit = line.split()
    current_trial_id = int(linesplit[-1])
    time = int(linesplit[1])
    return {"participant_id": participant_id
                    , "trial_id": current_trial_id
                    , "time": time
                    , "event": "TRIALID"
                    , "colour": None
                    , "coordinates": None}
    
def get_events_from_start_line(participant_id, current_trial_id, line):
    linesplit = line.split()
    time = int(linesplit[1])
    return {"participant_id": participant_id
                    , "trial_id": current_trial_id
                    , "time": time
                    , "event": "START"
                    , "colour": None
                    , "coordinates": None}

def get_events_from_end_line(participant_id, current_trial_id, line):
    linesplit = line.split()
    time = int(linesplit[1])
    return {"participant_id": participant_id
                    , "trial_id": current_trial_id
                    , "time": time
                    , "event": "END"
                    , "colour": None
                    , "coordinates": None}

def get_events_from_trial_var_data_line(participant_id, current_trial_id, line):
    linesplit = line.split()
    time = int(linesplit[1])
    stimulus_onset_in_seconds = float(line.split()[-1])
    stimulus_onset_in_milliseconds = int(stimulus_onset_in_seconds * 1000)
    coordinates = " ".join(linesplit[4:5])
    return {"participant_id": participant_id
                    , "trial_id": current_trial_id
                    , "time": time
                    , "event": "TRIAL_VAR_DATA"
                    , "colour": None
                    , "coordinates": coordinates
                    , "stimulus_onset_ms": stimulus_onset_in_milliseconds}


def get_events_from_fixpoint_line(participant_id, current_trial_id, line):
    linesplit = line.split()
    time = int(linesplit[1])
    colour = " ".join(linesplit[4:7])
    coordinates = " ".join(linesplit[10:12])
    return {"participant_id": participant_id
                    , "trial_id": current_trial_id
                    , "time": time
                    , "event": "FIXPOINT"
                    , "colour": colour
                    , "coordinates": coordinates}

def process_line(line, participant_id, current_trial_id):
    # Detect trial start
    if "TRIALID" in line:
        event = get_events_from_trial_id_line(participant_id, line)
        current_trial_id = event["trial_id"]
        return [event, current_trial_id]
    # Detect beginning of Trial
    elif line.startswith("START"):
        event = get_events_from_start_line(participant_id, current_trial_id, line)
        return [event]
    elif line.startswith("END"):
        event = get_events_from_end_line(participant_id, current_trial_id, line)
        return [event]
    elif "!V TRIAL_VAR_DATA" in line:
        event = get_events_from_trial_var_data_line(participant_id, current_trial_id, line)
        return [event]
    elif "FIXPOINT" in line:
        event = get_events_from_fixpoint_line(participant_id, current_trial_id, line)
        return [event]
    else:
        return [None]

def add_stimulus_events(events:pd.DataFrame):
    stimulus_onset_events = []
    red_fixpoints = events[(events['event'] == "FIXPOINT") & (events["colour"] == RED)]

    for _, row in red_fixpoints.iterrows():
        print(row)
        



def process_asc_file(filename):
    print(f"Processing {filename}")

    filepath = ASC_DIR / filename
    participant_id = filename.split("_")[1]

    with open(filepath, 'r') as fp:
        lines = fp.readlines()

    events = []
    current_trial_id = None

    for line in lines:
        line = line.strip()
        line_event = process_line(line=line,
                     participant_id=participant_id,
                     current_trial_id=current_trial_id)
        if line_event[0] is None:
            continue
        elif len(line_event) > 1:
            current_trial_id = line_event[1]
        events.append(line_event[0])

    events_df = pd.DataFrame(events)
    
    print("Finished initial event loading")
    print("Adding stimulus events")

    add_stimulus_events(events_df)




def main():
    #participant_metadata = pd.read_excel(DATA_DIR / "demographic_info.xlsx")
    asc_files = [f for f in os.listdir(ASC_DIR) if f.endswith('.asc')]
    process_asc_file(asc_files[1])



if __name__ == '__main__':
    main()

