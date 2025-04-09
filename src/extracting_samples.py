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
            "error_message"],
    "Standard": ["experiment",
            "participant_id",
            "trial_id",
            "time",
            "x_left",
            "y_left",
            "pupil_size_left",
            "x_velocity_left",
            "y_velocity_left",
            "x_right",
            "y_right",
            "pupil_size_right",
            "x_velocity_right",
            "y_velocity_right",
            "x_resolution",
            "y_resolution",
            "error_message"]
}

type_map = {
    "LR": {"time": "Int64",
            "x_left": "float64",
            "y_left": "float64",
            "pupil_size_left": "float64",
            "x_right": "float64",
            "y_right": "float64",
            "pupil_size_right": "float64",
            "x_velocity_left": "float64",
            "y_velocity_left": "float64",
            "x_velocity_right": "float64",
            "y_velocity_right": "float64",
            "x_resolution": "float64",
            "y_resolution": "float64",
            ".": "string",
            "error_message": "string"},
    "R": {"time": "Int64",
            "x_right": "float64",
            "y_right": "float64",
            "pupil_size_right": "float64",
            "x_velocity_right": "float64",
            "y_velocity_right": "float64",
            "x_resolution": "float64",
            "y_resolution": "float64",
            ".": "string",
            "error_message": "string"},
    "L": {"time": "Int64",
            "x_left": "float64",
            "y_left": "float64",
            "pupil_size_left": "float64",
            "x_velocity_left": "float64",
            "y_velocity_left": "float64",
            "x_resolution": "float64",
            "y_resolution": "float64",
            ".": "string",
            "error_message": "string"}
}

na_values ={
    "LR": {col: "." for col in list(set(colnames["LR"]) & set(numeric_cols))},
    "R": {col: "." for col in list(set(colnames["R"]) & set(numeric_cols))},
    "L": {col: "." for col in list(set(colnames["L"]) & set(numeric_cols))}
}

def set_column_dtype(df):
    print("Transform numeric columns\n")
    
    for col, dtype in type_map_samples.items():
        if col in df.columns:
            try:
                if ("float" in dtype) | ("int" in dtype):
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as e:
                print(f"Failed to convert {col} to {dtype}: {e}")
        else:
            print(f"Column {col} not found in DataFrame")
        
    return df

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
                    dtype = type_map[eyes_tracked])
    df = df.drop(".", axis=1)  
              
    for col in df.columns:
        if col not in [".", "error_message"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def process_asc_files(asc_files, experiment):
    print("Processing")
    df_events = pd.read_parquet(CLEANED_DIR / f"{experiment}_events.pq")
    df_events = df_events[~(df_events["event"] == "FIXPOINT")]
    
    path_save = CLEANED_DIR / f"{experiment}_samples.pq"
    first_write = True
    
    for file_name in tqdm(asc_files):
        file_path = ASC_RAW_SAMPLES_DIR / file_name
        print(file_path)
        # Information from event
        participant_id = file_name.split("_")[1]
        
        # Only process sample data, if participant also exists in event data
        if str(participant_id) not in df_events["participant_id"].unique():
            continue
        
        df_event = df_events[df_events["participant_id"]==f"{participant_id}"]
        
        eyes = df_event["eye"].dropna().unique() 
        if "L" in eyes and "R" in eyes:
            df = read_asc_file(file_path, "LR")
        elif "L" in eyes:
            df = read_asc_file(file_path, "L")
        elif "R" in eyes:
            df = read_asc_file(file_path, "R")
        else: continue
        
        # Add experiment, participant_id and trial_id from event file
        df = add_info_from_event(df, experiment, participant_id, df_event)        
        
        # Ensure all standard columns exist in the dataframe
        for col in colnames["Standard"]:
            if col not in df.columns:
                # Add missing columns with NaN values
                df[col] = pd.NA
        
        # Ensure columns are in the same order
        df = df[colnames["Standard"]]
        
        # Set data-types
        df = set_column_dtype(df)
        
        df.to_parquet(path_save, engine="fastparquet", append = not first_write)
        first_write = False
            
    return

def main(experiments, file_filters):
    # Convert asc files to parquet files
    for file_filter, experiment in zip(file_filters, experiments):
        asc_files = [f for f in os.listdir(ASC_RAW_SAMPLES_DIR) if f.endswith('.asc') and f.startswith(f"{file_filter}")]
        df = process_asc_files(asc_files, experiment=experiment)
        
        path_save = CLEANED_DIR / f"{experiment}_samples.pq"
        print(f"Saving to {path_save}")
        df.to_parquet(path_save, index=False)
        process_asc_files(asc_files, experiment=experiment)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract events from ASC files.")
    parser.add_argument("--experiments", nargs='+', required=True, help="List of experiment names")
    parser.add_argument("--file_filters", nargs='+', required=True, help="List of file filters")
    args = parser.parse_args()
    
    if len(args.experiments) != len(args.file_filters):
        raise ValueError("experiments and file_filters must be the same length")

    main(args.experiments, args.file_filters)