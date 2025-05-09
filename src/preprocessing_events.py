import pandas as pd
from config import *
import argparse
import logging
import numpy as np
from tqdm import tqdm


########################
###   ANTI_SACCADE   ###
########################

def coalesce_time_elapsed(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Coalesce time elapsed and delay")
    # Use delay as a fallback for time_elapsed and then drop the delay column
    return (
        df.assign(
            time_elapsed = lambda x: x[['time_elapsed', 'delay']].bfill(axis=1)['time_elapsed']
        )
        .drop(columns=['delay'])
    )

def fill_values_side(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Fill missing values for side columns")
    
    # Forward and backward fill the 'side' column within each participant/trial group
    return(
        df.sort_values(['participant_id', 'trial_id', 'time'])
        .groupby(['participant_id', 'trial_id'], group_keys=False)[df.columns]
        .apply(lambda g: g.assign(side=g['side'].ffill().bfill()))
    )
    
def stimulus_onset_time(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Calculate stimulus onset time")
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

def stimulus_active(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    logging.info("Set stimulus active")

    df = df.sort_values(["participant_id", "trial_id", "time"])

    if experiment in ["FITTS_LAW", "FIXATIONS", "EVIL_BASTARD", "SHAPES", "SMOOTH_PURSUITS"]:
        condition = df["event"] == "FIXPOINT"
    if experiment in ["ANTI_SACCADE", "REACTION"]:
        condition = (df["event"] == "FIXPOINT") & (df["colour"] == RED)
    
    df["stimulus_active"] = (
        df
        .assign(stimulus_active = np.where(condition, True, None))
        .groupby(["participant_id", "trial_id"])["stimulus_active"]
        .transform(lambda x: x.ffill().astype(bool).fillna(False))
    )
            
    return df

def preprocess_anti_saccade(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    logging.info("Preprocessing anti_saccade")
    df_trans = (df
        .pipe(coalesce_time)
        .pipe(set_column_dtype)
        .pipe(coalesce_time_elapsed)
        .pipe(fill_values_side)
        .pipe(stimulus_onset_time)
        .pipe(standardise_time)
        .pipe(fill_values, ["colour","stimulus_x", "stimulus_y"])
        .pipe(stimulus_active, experiment)
        .pipe(limit_x_points)
        .pipe(limit_y_points)
    )
    
    return df_trans

########################
###   EVIL_BASTARD   ###
########################

def preprocess_evil_bastard(df: pd.DataFrame, experiment:str) -> pd.DataFrame:
    logging.info("Preprocessing evil bastard")
    df_trans = (df
        .pipe(coalesce_time)
        .pipe(set_column_dtype)
        .pipe(fill_values, ["colour","stimulus_x", "stimulus_y"])
        .pipe(stimulus_active, experiment)
        .pipe(limit_x_points)
        .pipe(limit_y_points)
    )
    return df_trans



###############################
###   REACTION/PROSACCADE   ###
###############################

def coalesce_stimulus_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Coalesce stimulus coordinates")
    df.loc[:,"stimulus_x"] = df[["pos_x", "stimulus_x"]].bfill(axis=1)
    df.loc[:,"stimulus_y"] = df[["pos_y", "stimulus_y"]].bfill(axis=1)
    df = df.drop(["pos_x", "pos_y"], axis=1)
    return df

def preprocess_reaction(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    logging.info("Preprocessing reaction")
    df_trans = (df
        .pipe(coalesce_time)
        .pipe(set_column_dtype)
        .pipe(coalesce_stimulus_coordinates)
        .pipe(fill_values, ["colour","stimulus_x", "stimulus_y"])
        .pipe(stimulus_active, experiment)
        .pipe(limit_x_points)
        .pipe(limit_y_points)
    )
    return df_trans


#####################
###   FITTS_LAW   ###
#####################

def preprocess_fitts_law(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    logging.info("Preprocessing fitts law")
    df_trans = (df
        .pipe(coalesce_time)
        .pipe(set_column_dtype)
        .pipe(fill_values, ["distance", "target_width"])
        .pipe(stimulus_active, experiment)
        .pipe(limit_x_points)
        .pipe(limit_y_points)
    )
    return df_trans


#######################
###   KING_DEVICK   ###
#######################

def preprocess_king_devick(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    logging.info("Preprocessing king devick")
    df_trans = (df
        .pipe(coalesce_time)
        .pipe(set_column_dtype)
        .pipe(fill_values, ["marks", "time_elapsed"], backfill=True)
        .pipe(limit_x_points)
        .pipe(limit_y_points)
    )
    return df_trans


############
## SHAPES ##
############


def preprocess_shapes(df: pd.DataFrame, experiment:str) -> pd.DataFrame:
    logging.info("Preprocessing shapes")
    df_trans = (df
        .pipe(coalesce_time)
        .pipe(set_column_dtype)
        .pipe(fill_values, ["shape"], backfill=True)
        .pipe(stimulus_active, experiment)
        .pipe(limit_x_points)
        .pipe(limit_y_points)
    )
    return df_trans



####################
## SMOOTH PURSUIT ##
####################

def preprocess_smooth_pursuit(df: pd.DataFrame, experiment:str) -> pd.DataFrame:
    logging.info("Preprocessing smooth pursuits")
    df_trans = (df
        .pipe(coalesce_time)
        .pipe(set_column_dtype)
        .pipe(fill_values, ["shape", "speed"], backfill=True)
        .pipe(stimulus_active, experiment)
        .pipe(limit_x_points)
        .pipe(limit_y_points)
    )
    return df_trans



#####################
###   Fixations   ###
#####################

def preprocess_fixations(df: pd.DataFrame, experiment:str) -> pd.DataFrame:
    logging.info("Preprocessing fixations")
    df_trans = (df
        .pipe(coalesce_time)
        .pipe(set_column_dtype)
        .pipe(coalesce_stimulus_coordinates)
        .pipe(fill_values, ["target_shape", "stimulus_x", "stimulus_y"])
        .pipe(stimulus_active, experiment)
        .pipe(limit_x_points)
        .pipe(limit_y_points)
    )
    
    return df_trans


###################
###   General   ###
###################

def limit_x_points(df: pd.DataFrame, columns: list = ["stimulus_x", "pos_x", "x", "start_x", "end_x"]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df.loc[:,col] = df[col].clip(lower=0, upper=1920)
    return df


def limit_y_points(df: pd.DataFrame, columns: list = ["stimulus_y", "pos_y", "y", "start_y", "end_y"]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df.loc[:,col] = df[col].clip(lower=0, upper=1080)
    return df

def set_column_dtype(df: pd.DataFrame) -> pd.DataFrame:
    logging.info(f"Starting dtype transformation")
    
    for col, dtype in type_map.items():
        if col in df.columns:
            try:
                if ("float" in dtype) | ("int" in dtype):
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as e:
                logging.error(f"Failed to convert '{col}' to {dtype}: {e}")
        else:
            logging.warning(f"Column '{col}' not found in DataFrame - this is probably okay")
    logging.info(f"Finished dtype transformation")

    return df
                

def coalesce_time(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Coalesce time")
    df.loc[:,"time"] = df[["time", "start_time"]].bfill(axis=1)
    
    return df

def group_df(df: pd.DataFrame) -> pd.DataFrame:
    grouped_df = df.sort_values(["participant_id", "trial_id", "time"]).groupby(["participant_id", "trial_id"])
    
    return grouped_df

def remove_trialid_event(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Remove trial id event")
    df = df[df["event"] != "TRIALID"]
    
    return df

def standardise_time(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Standardise time")
    
    grouped_df = group_df(df)
    
    df.loc[:,"time"] = df["time"] - grouped_df.time.transform('min').astype("float64")
    df.loc[:,"start_time"] = df["start_time"] - grouped_df.time.transform('min').astype("float64")
    df.loc[:,"end_time"] = df["end_time"] - grouped_df.time.transform('min').astype("float64")
    
    return df

def fill_values(df: pd.DataFrame, fill_on_columns: list, forwardfill:bool = True, backfill:bool = False) -> pd.DataFrame:
    logging.info("Fill missing values")
    grouped_df = group_df(df)
    for col in fill_on_columns:
        if forwardfill and backfill:
            df.loc[:,col] = grouped_df[col].bfill().ffill()
        elif backfill:
            df.loc[:,col] = grouped_df[col].bfill()
        elif forwardfill:
            df.loc[:,col] = grouped_df[col].ffill()
    return df

def remove_invalid_saccades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove ESACC events that have a blink (SBLINK + EBLINK) during the saccade (SSACC to ESACC).
    """
    logging.info("Remove invalid saccades")
    
    df_sorted = df.sort_values(["participant_id", "trial_id", "time"]).reset_index()  # keep original index
    keep_original_indices = []

    for (participant, trial), group in tqdm(df_sorted.groupby(["participant_id", "trial_id"], sort=False)):
        group = group.reset_index(drop=True)
        i = 0
        while i < len(group):
            event = group.loc[i, 'event']
            if event == 'SSACC':
                start_idx = i
                sblink_present = False
                eblink_present = False
                i += 1
                found_esacc = False
                while i < len(group):
                    curr_event = group.loc[i, 'event']
                    if curr_event == 'SBLINK':
                        sblink_present = True
                    elif curr_event == 'EBLINK':
                        eblink_present = True
                    elif curr_event == 'ESACC':
                        found_esacc = True
                        if not (sblink_present and eblink_present):
                            keep_original_indices.extend(group.loc[start_idx:i+1, 'index'].tolist())
                        i += 1  # move past the ESACC
                        break
                    i += 1
                if not found_esacc:
                    # If we never found a matching ESACC, just move on
                    i = start_idx + 1
            else:
                keep_original_indices.append(group.loc[i, 'index'])
                i += 1

    # Drop duplicates in case anything got added twice
    keep_original_indices = list(dict.fromkeys(keep_original_indices))

    return df_sorted[df_sorted["index"].isin(keep_original_indices)].sort_values(["participant_id", "trial_id", "time"]).drop(["index"],axis=1).reset_index(drop=True)


def remove_blink_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove blink events, as they are only used to remove invalid saccades.
    """
    logging.info("Remove start events")
    
    mask = (df["event"] == "SBLINK") | (df["event"] == "EBLINK")
    df_masked = df.loc[~mask,:]
    
    return df_masked

def remove_start_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove SFIX and SSACC events.
    
    SFIX and SSACC event does not contain other information than the start time
    which is also included in EFIX and ESACC.
    """
    logging.info("Remove start events")
    
    mask = (df["event"] == "SFIX") | (df["event"] == "SSACC")
    df_masked = df.loc[~mask,:]
    
    return df_masked

### RUN

def preprocess_general(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Performing general preprocessing")
    
    df_transformed = (df
        .pipe(remove_trialid_event)
        .pipe(standardise_time)
        .pipe(remove_invalid_saccades)
        .pipe(remove_blink_events)
        .pipe(remove_start_events)
        
        # This should be last
        .reset_index(drop=True)
    )
    
    return df_transformed

def preprocess_experiment(experiment:str) -> None:
    df = pd.read_parquet(f"{CLEANED_DIR}/{experiment}_events.pq")
    
    preprocessing_funcs = {
        "ANTI_SACCADE": preprocess_anti_saccade,
        "REACTION": preprocess_reaction,
        "FITTS_LAW": preprocess_fitts_law,
        "KING_DEVICK": preprocess_king_devick,
        "EVIL_BASTARD": preprocess_evil_bastard,
        "SHAPES": preprocess_shapes,
        "SMOOTH_PURSUITS": preprocess_smooth_pursuit,
        "FIXATIONS": preprocess_fixations
    }

    df = preprocessing_funcs.get(experiment, lambda x,y: x)(df, experiment)
    
    # General preprocessing 
    df_transformed = preprocess_general(df)
    
    # Save to parquet
    df_transformed.to_parquet(PREPROCESSED_DIR / f"{experiment}_events.pq")

def main(experiments: list[str]):
    # Convert asc files to parquet files
    for experiment in experiments:
        logging.info(f"Starting preprocessing process of experiment: {experiment}")
        preprocess_experiment(experiment)
        logging.info(f"Finished preprocessing process of experiment: {experiment}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract events from ASC files.")
    parser.add_argument("--experiments", nargs='+', required=True, help="List of experiment names")
    args = parser.parse_args()
    
    main(args.experiments)