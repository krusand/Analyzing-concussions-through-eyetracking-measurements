import pandas as pd
from config import *

experiments = ["ANTI_SACCADE"] #, "FITTS_LAW", "FIXATIONS", "KING_DEVICK", "EVIL_BASTARD", "REACTION", "SHAPES", "SMOOTH_PURSUITS"]

########################
###   ANTI_SACCADE   ###
########################

def transform_numeric_columns(df):
    for col in numeric_columns_anti_saccade:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df

def coalesce_time_elapsed(df):
    return (
        df.assign(
            time_elapsed = lambda x: x[['time_elapsed', 'delay']].bfill(axis=1)['time_elapsed']
        )
        .drop(columns=['delay'])
    )

def fill_values_side(df):
    return(
        df.sort_values(['participant_id', 'trial_id', 'time'])
        .groupby(['participant_id', 'trial_id'], group_keys=False)[df.columns]
        .apply(lambda g: g.assign(side=g['side'].ffill().bfill()))
    )

def stimulus_onset_time(df):
    return (
        df.sort_values(['participant_id', 'trial_id', 'time'])
        .groupby(['participant_id', 'trial_id'], group_keys=False)[df.columns]
        .apply(lambda g: g.assign(
                time = g.apply(lambda row: 
                    row['time'] if row['colour'] != '255 0 0' 
                        else (g.loc[(g['colour'] == '255 255 255'), 'time'].iloc[0] + 1000 * g.loc[(g['event'] == 'TRIAL_VAR_DATA'), 'time_elapsed'].iloc[0]), 
                    axis=1))
            )
    )


def preprocess_anti_saccade(df):
    df_trans = (
        df.pipe(transform_numeric_columns)
        .pipe(coalesce_time_elapsed)
        .pipe(fill_values_side)
        .pipe(stimulus_onset_time)
    )
    
    return df_trans

###################
###   General   ###
###################

def coalesce_time(df):
    df.loc[:,"time"] = df[["time", "end_time"]].bfill(axis=1)
    
    return df
    
def remove_start_events(df):
    mask = (df["event"] == "SFIX") | (df["event"] == "SSACC")
    df_masked = df.loc[~mask,:]
    
    return df_masked

def group_df(df):
    grouped_df = df.sort_values(["participant_id", "trial_id", "time"]).groupby(["participant_id", "trial_id"])#[df.columns]
    
    return grouped_df

def standardise_time(df):
    
    grouped_df = group_df(df)
    
    df.loc[:,"time"] = df["time"] - grouped_df.time.transform('min')
    df.loc[:,"start_time"] = df["start_time"] - grouped_df.time.transform('min')
    df.loc[:,"end_time"] = df["end_time"] - grouped_df.time.transform('min')
    
    return df

def fill_values(df):
    
    grouped_df = group_df(df)
    df.loc[:,"colour"] = grouped_df["colour"].ffill()
    df.loc[:,"stimulus_x"] = grouped_df["stimulus_x"].ffill()
    df.loc[:,"stimulus_y"] = grouped_df["stimulus_y"].ffill()
    
    return df

def preprocess_general(df):
    df_transformed = (
        df.pipe(remove_start_events)
        .pipe(coalesce_time)
        .pipe(standardise_time)
        .pipe(fill_values)
    )
    
    return df_transformed

def preprocess_experiment(experiment):
    df = pd.read_parquet(f"{CLEANED_DIR}/{experiment}.pq")
    
    # Preprocessing specific for each event
    if experiment == "ANTI_SACCADE":
        df = preprocess_anti_saccade(df)
    
    # General preprocessing 
    df_transformed = preprocess_general(df)
    
    # Save to parquet
    df_transformed.to_parquet(PREPROCESSED_DIR / f"{experiment}.pq")

def preprocess():
    
    for experiment in experiments:
        preprocess_experiment(experiment)

def main():
    # Convert asc files to parquet files
    preprocess()

if __name__ == '__main__':
    main()