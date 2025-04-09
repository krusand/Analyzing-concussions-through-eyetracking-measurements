import pandas as pd
from config import *
import argparse

import numpy as np

experiments = ["ANTI_SACCADE"] #, "FITTS_LAW", "FIXATIONS", "KING_DEVICK", "EVIL_BASTARD", "REACTION", "SHAPES", "SMOOTH_PURSUITS"]

from tqdm import tqdm


########################
###   ANTI_SACCADE   ###
########################



def set_column_dtype_anti_saccade(df):
    print("Transform numeric columns\n")
    
    for col, dtype in type_map["ANTI_SACCADE"].items():
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
                
def coalesce_time_elapsed(df):
    print("Coalesce time elapsed and delay\n")
    # Use delay as a fallback for time_elapsed and then drop the delay column
    return (
        df.assign(
            time_elapsed = lambda x: x[['time_elapsed', 'delay']].bfill(axis=1)['time_elapsed']
        )
        .drop(columns=['delay'])
    )

def fill_values_side(df):
    print("Fill missing values for side columns\n")
    
    # Forward and backward fill the 'side' column within each participant/trial group
    return(
        df.sort_values(['participant_id', 'trial_id', 'time'])
        .groupby(['participant_id', 'trial_id'], group_keys=False)[df.columns]
        .apply(lambda g: g.assign(side=g['side'].ffill().bfill()))
    )
    
def stimulus_onset_time(df):
    print("Calculate stimulus onset time\n")
    results = []
    
    # Process each group separately
    for (participant, trial), group in df.sort_values(['participant_id', 'trial_id', 'time']).groupby(['participant_id', 'trial_id']):
        # Create a mask for red color
        red_mask = group['colour'] == '255 0 0'
        
        if red_mask.any():
            # Get base time and time elapsed once per group
            try:
                base_time = group.loc[group['colour'] == '255 255 255', 'time'].iloc[0]
                time_factor = group.loc[group['event'] == 'TRIAL_VAR_DATA', 'time_elapsed'].iloc[0]
                
                # Calculate new time for all red entries at once
                new_time = int(base_time + 1000 * time_factor)
                
                # Create a copy of the group's time column
                new_times = group['time'].copy()
                
                # Update only the red entries
                new_times.loc[red_mask] = new_time
                
                # Convert dtype
                new_times = new_times.astype("Int64")
                
                # Assign the modified time back to the group
                group = group.assign(time=new_times)
            except IndexError:
                # Handle case where required data is missing
                pass
        
        results.append(group)
    
    return pd.concat(results) if results else df.copy()

def fill_values(df):
    
    grouped_df = group_df(df)
    df.loc[:,"colour"] = grouped_df["colour"].ffill()
    df.loc[:,"stimulus_x"] = grouped_df["stimulus_x"].ffill()
    df.loc[:,"stimulus_y"] = grouped_df["stimulus_y"].ffill()
    
    return df

def stimulus_active(df):
    print("Set stimulus active")
    df = df.assign(
        stimulus_active = df["colour"]
            .map({
                "255 0 0": True,
                "255 255 255": False
            }
        )
    )
    return df


def preprocess_anti_saccade(df):
    print("Preprocessing: Antisaccade:")
    df_trans = (
        df.pipe(coalesce_time)
        .pipe(set_column_dtype_anti_saccade)
        .pipe(coalesce_time_elapsed)
        .pipe(fill_values_side)
        .pipe(stimulus_onset_time)
        .pipe(standardise_time)
        .pipe(fill_values)
        .pipe(stimulus_active)
    )
    
    return df_trans

########################
###   EVIL_BASTARD   ###
########################

def set_column_dtype_evil_bastard(df):
    print("Transform numeric columns\n")
    
    for col, dtype in type_map["EVIL_BASTARD"].items():
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

def preprocess_evil_bastard(df):
    df_trans = (df
        .pipe(coalesce_time)
        .pipe(set_column_dtype_evil_bastard)
        .pipe(fill_values)
    )
    return df_trans


###################
###   General   ###
###################

def coalesce_time(df):
    print("Coalesce time\n")
    df.loc[:,"time"] = df[["time", "start_time"]].bfill(axis=1)
    
    return df

def group_df(df):
    grouped_df = df.sort_values(["participant_id", "trial_id", "time"]).groupby(["participant_id", "trial_id"])#[df.columns]
    
    return grouped_df

def remove_trialid_event(df):
    print("Remove trial id event\n")
    df = df[df["event"] != "TRIALID"]
    
    return df

def standardise_time(df):
    print("Standardise time\n")
    
    grouped_df = group_df(df)
    
    df.loc[:,"time"] = df["time"] - grouped_df.time.transform('min').astype("float64")
    df.loc[:,"start_time"] = df["start_time"] - grouped_df.time.transform('min').astype("float64")
    df.loc[:,"end_time"] = df["end_time"] - grouped_df.time.transform('min').astype("float64")
    
    return df

def fill_values(df):
    print("Fill missing values\n")
    
    grouped_df = group_df(df)
    df.loc[:,"colour"] = grouped_df["colour"].ffill()
    df.loc[:,"stimulus_x"] = grouped_df["stimulus_x"].ffill()
    df.loc[:,"stimulus_y"] = grouped_df["stimulus_y"].ffill()
    
    return df

def remove_invalid_saccades(df):
    """
    Remove ESACC events that have a blink occurring during the saccade.
    
    A saccade is considered invalid if there is a SBLINK and EBLINK event
    between the corresponding SSACC and ESACC events.
    """
    print("Remove invalid saccades\n")
    # Sort the data by participant_id, trial_id, and time
    df_sorted = df.sort_values(["participant_id", "trial_id", "time"])
    
    # Process each participant and trial separately
    results = []
    
    for (participant, trial), group in tqdm(df_sorted.groupby(["participant_id", "trial_id"])):
        # Reset index to iterate through rows sequentially
        group = group.reset_index(drop=True)
        rows_to_keep = []
        
        # Track current saccade start
        in_saccade = False
        has_sblink = False
        has_eblink = False
        
        for i, row in group.iterrows():
            event = row['event']
            
            # Start of a new saccade
            if event == 'SSACC':
                in_saccade = True
                has_sblink = False
                has_eblink = False
                rows_to_keep.append(row)
            
            # During a saccade, track blinks
            elif in_saccade:
                if event == 'SBLINK':
                    has_sblink = True
                    rows_to_keep.append(row)
                elif event == 'EBLINK':
                    has_eblink = True
                    rows_to_keep.append(row)
                elif event == 'ESACC':
                    # Only keep ESACC if there wasn't a complete blink during this saccade
                    if not (has_sblink and has_eblink):
                        rows_to_keep.append(row)
                    in_saccade = False
                else:
                    # Keep all other events
                    rows_to_keep.append(row)
            
            # Not in a saccade, keep everything
            else:
                rows_to_keep.append(row)
        
        # Create a dataframe from the kept rows
        if rows_to_keep:
            results.append(pd.DataFrame(rows_to_keep))
    
    # Combine all results
    return pd.concat(results) if results else pd.DataFrame(columns=df.columns)

def remove_start_events(df):
    """
    Remove SFIX and SSACC events.
    
    SFIX and SSACC event does not contain other information than the start time
    which is also included in EFIX and ESACC.
    """
    print("Remove start events\n")
    
    mask = (df["event"] == "SFIX") | (df["event"] == "SSACC")
    df_masked = df.loc[~mask,:]
    
    return df_masked

def preprocess_general(df):
    print("Preprocess: General:")
    
    df_transformed = (
        df
        .pipe(remove_trialid_event)
        .pipe(standardise_time)
        .pipe(remove_invalid_saccades)
        .pipe(remove_start_events)
    )
    
    return df_transformed

def preprocess_experiment(experiment):
    df = pd.read_parquet(f"{CLEANED_DIR}/{experiment}_events.pq")
    
    # Preprocessing specific for each event
    if experiment == "ANTI_SACCADE":
        df_transformed = preprocess_anti_saccade(df)
    elif experiment == "EVIL_BASTARD":
        df = preprocess_evil_bastard(df)
    
    # General preprocessing 
    df_transformed = preprocess_general(df)
    
    # Save to parquet
    df_transformed.to_parquet(PREPROCESSED_DIR / f"{experiment}_events.pq")

def main(experiments):
    # Convert asc files to parquet files
    for experiment in experiments:
        preprocess_experiment(experiment)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract events from ASC files.")
    parser.add_argument("--experiments", nargs='+', required=True, help="List of experiment names")
    args = parser.parse_args()
    
    main(args.experiments)