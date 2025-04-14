import pandas as pd
from config import *
import argparse
SPECIAL_PARTICIPANTS = ["87", "89", "93", "96", "103", "105", "109", "117", "118", "119", "120", "127", "128", "141"]

def print_info_removed_rows(filtered_df, df):
    logging.info("Removed the following rows [p_id, t_id]")
    removed_trials = df[["participant_id", "trial_id"]].drop_duplicates()
    kept_trials = filtered_df[["participant_id", "trial_id"]].drop_duplicates()

    removed_ids = pd.merge(removed_trials, kept_trials, on=["participant_id", "trial_id"], how="outer", indicator=True)
    removed_ids = removed_ids.query('_merge == "left_only"').drop(columns="_merge")

    if len(removed_ids) > 0:
        print(removed_ids.to_numpy())

def exclude_nan_participants(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Removing na participants")

    filtered_df = df[df["participant_id"].notna()]
    return filtered_df


def exclude_special_participants(df: pd.DataFrame, special_participants: list[str]) -> pd.DataFrame:
    logging.info("Removing special participants")

    filtered_df = df[~df["participant_id"].isin(special_participants)]
    return filtered_df

def remove_invalid_fixpoints(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Removing invalid fixpoints")
    experiment = df["experiment"].unique()[0]
    if experiment == 'ANTI_SACCADE':
        df[['stimulus_x', 'stimulus_y']] = df[['stimulus_x', 'stimulus_y']].apply(pd.to_numeric, errors='coerce')
        filtered_df = df[
             (df['stimulus_x'].between(0, 1920)) | (df['stimulus_x'].isna()) &
             (df['stimulus_y'].between(0, 1080)) | (df['stimulus_y'].isna())
        ]
    elif experiment == "EVIL_BASTARD":
        filtered_df = df.loc[~(df["colour"]==BLUE),:]
    elif experiment == "REACTION":
        df[['pos_x', 'pos_y', 'stimulus_x', 'stimulus_y']] = df[['pos_x', 'pos_y', 'stimulus_x', 'stimulus_y']].apply(pd.to_numeric, errors='coerce')
        filtered_df = df[
             (df['pos_x'].between(0, 1920)) | (df['pos_x'].isna()) &
             (df['stimulus_x'].between(0, 1920)) | (df['stimulus_x'].isna()) &
             (df['pos_y'].between(0, 1080)) | (df['pos_y'].isna()) &
             (df['stimulus_y'].between(0, 1080)) | (df['stimulus_y'].isna())
        ]
    else:
        logging.info("Removed 0 rows")
        return df
    
    rows_removed = len(df)-len(filtered_df)
    
    logging.info(f"Removed {rows_removed} rows")
    if rows_removed > 0:
        print_info_removed_rows(filtered_df, df)
    return filtered_df

def check_trialid_event(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Checking if there is a trial_id")
    df_check = (df.
        query("event == 'TRIALID'").
        groupby(["participant_id", "trial_id", "event"])["event"].
        count().
        reset_index(name='n_TRIALID').
        query("n_TRIALID == 1")
        [["participant_id", "trial_id"]]
    )
    
    filtered_df = pd.merge(df, df_check, how='inner', on = ["participant_id", 'trial_id'])
     
    rows_removed = len(df)-len(filtered_df)
    
    logging.info(f"Removed {rows_removed} rows")
    if rows_removed > 0:
        print_info_removed_rows(filtered_df, df)
    
    return filtered_df

def check_fixpoint_amount(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Checking if there are the correct amount of fixpoints for the given experiment")
    
    experiment = df["experiment"].unique()[0]
    if experiment == "ANTI_SACCADE":
        query = "n_fixpoints == 2"
    elif experiment == "EVIL_BASTARD":
        query = "n_fixpoints > 0"
    elif experiment == "REACTION":
        query = "n_fixpoints == 2"
    else:
        logging.info("Removed 0 rows")
        return df   
    
    df_check = (df.
        query("event == 'FIXPOINT'").
        groupby(["participant_id", "trial_id", "event"])["event"].
        count().
        reset_index(name='n_fixpoints').
        query(query)
        [["participant_id", "trial_id"]]
    ) 
    
    filtered_df = pd.merge(df, df_check, how='inner', on = ["participant_id", 'trial_id'])

    rows_removed = len(df)-len(filtered_df)
    
    logging.info(f"Removed {rows_removed} rows")
    if rows_removed > 0:
        print_info_removed_rows(filtered_df, df)
    return filtered_df

def check_red_fixpoint(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Checking if there are the correct amount of red fixpoints for the given experiment")
    experiment = df["experiment"].unique()[0]
    if experiment == "ANTI_SACCADE":
        df_check = (df.
            query("event == 'FIXPOINT' & colour == '255 0 0'").
            groupby(["participant_id", "trial_id", "event"])["event"].
            count().
            reset_index(name='n_red_fixpoints').
            query("n_red_fixpoints == 1")
        [["participant_id", "trial_id"]]
        )    
    else:
        logging.info("Removed 0 rows")
        return df
    
    filtered_df = pd.merge(df, df_check, how='inner', on = ["participant_id", 'trial_id'])

    
    rows_removed = len(df)-len(filtered_df)
    
    logging.info(f"Removed {rows_removed} rows")
    if rows_removed > 0:
        print_info_removed_rows(filtered_df, df)
    return filtered_df

def check_white_fixpoint(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Checking if there are the correct amount of white fixpoints for the given experiment")
    experiment = df["experiment"].unique()[0]
    if experiment == "ANTI_SACCADE":
        df_check = (df.
            query("event == 'FIXPOINT' & colour == '255 255 255'").
            groupby(["participant_id", "trial_id", "event"])["event"].
            count().
            reset_index(name='n_white_fixpoints').
            query("n_white_fixpoints == 1")
        [["participant_id", "trial_id"]]
        )
    else:
        logging.info("Removed 0 rows")
        return df
    
    filtered_df = pd.merge(df, df_check, how='inner', on = ["participant_id", 'trial_id'])

    rows_removed = len(df)-len(filtered_df)
    
    logging.info(f"Removed {rows_removed} rows")
    if rows_removed > 0:
        print_info_removed_rows(filtered_df, df)
    return filtered_df

def check_trial_var_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Checking if there are the correct amount of trial_var_data events for the given experiment")

    df_check = (df.
        query("event == 'TRIAL_VAR_DATA'").
        groupby(["participant_id", "trial_id", "event"])["event"].
        count().
        reset_index(name='n_trial_var_data_events').
        query("n_trial_var_data_events == 1")
        [["participant_id", "trial_id"]]
    )
    filtered_df = pd.merge(df, df_check, how='inner', on = ["participant_id", 'trial_id'])
       
    rows_removed = len(df)-len(filtered_df)
    
    logging.info(f"Removed {rows_removed} rows")
    if rows_removed > 0:
        print_info_removed_rows(filtered_df, df)
    return filtered_df

def check_start_event(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Checking if there are the correct amount of start events for the given experiment")

    df_check = (df.
        query("event == 'START'").
        groupby(["participant_id", "trial_id", "event"])["event"].
        count().
        reset_index(name='n_start').
        query("n_start == 1")
        [["participant_id", "trial_id"]]
    )
    
    filtered_df = pd.merge(df, df_check, how='inner', on = ["participant_id", 'trial_id'])

    rows_removed = len(df)-len(filtered_df)
 
    logging.info(f"Removed {rows_removed} rows")
    if rows_removed > 0:
        print_info_removed_rows(filtered_df, df)
    return filtered_df

def check_end_event(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Checking if there are the correct amount of end events for the given experiment")

    df_check = (df.
        query("event == 'END'").
        groupby(["participant_id", "trial_id", "event"])["event"].
        count().
        reset_index(name='n_end').
        query("n_end == 1")
        [["participant_id", "trial_id"]]
    )
    filtered_df = pd.merge(df, df_check, how='inner', on = ["participant_id", 'trial_id'])

    rows_removed = len(df)-len(filtered_df)

    logging.info(f"Removed {rows_removed} rows")
    if rows_removed > 0:
        print_info_removed_rows(filtered_df, df)
    return filtered_df




def clean_events(df:pd.DataFrame) -> pd.DataFrame:
    logging.info("Starting event cleaning")
    cleaned_df = (df.
        pipe(exclude_nan_participants).
        pipe(exclude_special_participants, special_participants=SPECIAL_PARTICIPANTS).
        pipe(remove_invalid_fixpoints).
        pipe(check_trialid_event).
        pipe(check_fixpoint_amount).
        pipe(check_red_fixpoint).
        pipe(check_white_fixpoint).
        pipe(check_trial_var_data).
        pipe(check_start_event).
        pipe(check_end_event)
    )
    return cleaned_df
    

def main(experiments):
    for experiment in experiments:
        df = pd.read_parquet(RAW_DIR / f"{experiment}_events.pq")
        cleaned_df = clean_events(df)
        cleaned_df.to_parquet(CLEANED_DIR / f"{experiment}_events.pq")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract events from ASC files.")
    parser.add_argument("--experiments", nargs='+', required=True, help="List of experiment names")
    args = parser.parse_args()
    
    main(args.experiments)





