import pandas as pd
import os
from tqdm import tqdm
from config import *
import argparse

numeric_cols = [
        "time", 
        "x_left", 
        "y_left", 
        "pupil_size_left", 
        "x_right", 
        "y_right", 
        "pupil_size_right", 
        "x_velocity_left", 
        "y_velocity_left", 
        "x_velocity_right", 
        "y_velocity_right", 
        "x_resolution", 
        "y_resolution"]

colnames ={
    "LR": ["time",
            "x_left",
            "y_left",
            "pupil_size_left",
            "x_right",
            "y_right",
            "pupil_size_right",
            "x_velocity_left",
            "y_velocity_left",
            "x_velocity_right",
            "y_velocity_right",
            "x_resolution",
            "y_resolution",
            ".",
            "error_message"],
    "R": ["time",
            "x_right",
            "y_right",
            "pupil_size_right",
            "x_velocity_right",
            "y_velocity_right",
            "x_resolution",
            "y_resolution",
            ".",
            "error_message"],
    "L": ["time",
            "x_left",
            "y_left",
            "pupil_size_left",
            "x_velocity_left",
            "y_velocity_left",
            "x_resolution",
            "y_resolution",
            ".",
            "error_message"]
}

dtypes = {
    "LR": {"time": int,
            "x_left": float,
            "y_left": float,
            "pupil_size_left": float,
            "x_right": float,
            "y_right": float,
            "pupil_size_right": float,
            "x_velocity_left": float,
            "y_velocity_left": float,
            "x_velocity_right": float,
            "y_velocity_right": float,
            "x_resolution": float,
            "y_resolution": float,
            ".": str,
            "error_message": str},
    "R": {"time": int,
            "x_right": float,
            "y_right": float,
            "pupil_size_right": float,
            "x_velocity_right": float,
            "y_velocity_right": float,
            "x_resolution": float,
            "y_resolution": float,
            ".": str,
            "error_message": str},
    "L": {"time": int,
            "x_left": float,
            "y_left": float,
            "pupil_size_left": float,
            "x_velocity_left": float,
            "y_velocity_left": float,
            "x_resolution": float,
            "y_resolution": float,
            ".": str,
            "error_message": str}
}

na_values ={
    "LR": {col: "." for col in list(set(colnames["LR"]) & set(numeric_cols))},
    "R": {col: "." for col in list(set(colnames["R"]) & set(numeric_cols))},
    "L": {col: "." for col in list(set(colnames["L"]) & set(numeric_cols))}
}

def add_info_from_event(df, experiment, participant_id, df_event):
    # Insert trial_id
    df_event_trials = df_event[(df_event["event"]=="START") | (df_event["event"]=="END")].loc[:,["trial_id", "time", "event"]]
    df_trials = df_event_trials.pivot(index="trial_id",columns="event")
    df_trials.columns = ["end_time", "start_time"]
    
    df.insert(loc=0, column="trial_id", value= "")
    for t_id, (start_time, end_time) in enumerate(zip(df_trials["start_time"], df_trials["end_time"])):
        df.loc[(df["time"] >= start_time) & (df["time"] <= end_time),"trial_id"] = t_id
    
    # Insert participant_id
    df.insert(loc=0, column='participant_id', value=participant_id)
    
    # Inser experiment
    df.insert(loc=0, column='experiment', value=experiment)
    
    return df

def read_asc_file(file_path, eyes_tracked):
    df = pd.read_csv(
                    file_path, 
                    names=colnames[eyes_tracked], 
                    delimiter="\t", 
                    skipinitialspace=True,
                    na_values=na_values[eyes_tracked],
                    dtype = dtypes[eyes_tracked])
    df = df.drop(".", axis=1)  
              
    for col in df.columns:
        if col not in [".", "error_message"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def process_asc_files(asc_files, experiment):
    samples = []
    df_events = pd.read_parquet(CLEANED_DIR / f"{experiment}_events.pq")
    df_events = df_events[~(df_events["event"] == "FIXPOINT")]
    
    for file_name in tqdm(asc_files):
        file_path = ASC_RAW_SAMPLES_DIR / file_name
        print(file_path)
        
        # Information from event
        participant_id = file_name.split("_")[1]
        if str(participant_id) in list(df_events["participant_id"]):
            df_event = df_events[df_events["participant_id"]==f"{participant_id}"]
        # Only process sample data, if participant also exists in event data
        else: continue
        
        if "L" in df_event["eye"].unique() and "R" in df_event["eye"].unique():
            df = read_asc_file(file_path, "LR")
        elif "L" in df_event["eye"].unique():
            df = read_asc_file(file_path, "L")
        elif "R" in df_event["eye"].unique():
            df = read_asc_file(file_path, "R")
        else: continue
        
        # Add experiment, participant_id and trial_id from event file
        df = add_info_from_event(df, experiment, participant_id, df_event)        
        
        samples.append(df)
            
    return pd.concat(samples, ignore_index=True)

def main(experiments, file_filters):
    # Convert asc files to parquet files
    for file_filter, experiment in zip(file_filters, experiments):
        asc_files = [f for f in os.listdir(ASC_RAW_SAMPLES_DIR) if f.endswith('.asc') and f.startswith(f"{file_filter}")]
        df = process_asc_files(asc_files, experiment=experiment)
        
        path_save = CLEANED_DIR / f"{experiment}_SAMPLES.pq"
        print(f"Saving to {path_save}")
        df.to_parquet(path_save, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract events from ASC files.")
    parser.add_argument("--experiments", nargs='+', required=True, help="List of experiment names")
    parser.add_argument("--file_filters", nargs='+', required=True, help="List of file filters")
    args = parser.parse_args()
    
    if len(args.experiments) != len(args.file_filters):
        raise ValueError("experiments and file_filters must be the same length")

    main(args.experiments, args.file_filters)