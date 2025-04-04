import pandas as pd
from config import *
SPECIAL_PARTICIPANTS = ["87", "89", "93", "96", "103", "105", "109", "117", "118", "119", "120", "127", "128", "141"]


def exclude_nan_participants(df: pd.DataFrame) -> pd.DataFrame:
    print("Removing na participants")
    print()

    filtered_df = df[df["participant_id"].notna()]
    return filtered_df


def exclude_special_participants(df: pd.DataFrame, special_participants: list[str]) -> pd.DataFrame:
    print("Removing special participants")
    print()

    filtered_df = df[~df["participant_id"].isin(special_participants)]
    return filtered_df

def check_trialid_event(df: pd.DataFrame) -> pd.DataFrame:
    print("Checking if there is a trial_id")
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
    
    print("Removed", rows_removed, "rows")
    if rows_removed > 0:
        print("Removed rows with [p_id, t_id]:\n", pd.unique(pd.concat([filtered_df,df]).drop_duplicates(keep=False)[["participant_id", "trial_id"]].values.ravel("K")))
    print()
    return filtered_df

def check_fixpoint_amount(df: pd.DataFrame) -> pd.DataFrame:
    print("Checking if there are the correct amount of fixpoints for the given experiment")
    experiment = df["experiment"].unique()[0]
    if experiment == "ANTI_SACCADE":
        df_check = (df.
            query("event == 'FIXPOINT'").
            groupby(["participant_id", "trial_id", "event"])["event"].
            count().
            reset_index(name='n_fixpoints').
            query("n_fixpoints == 2")
            [["participant_id", "trial_id"]]
        )
    else:
        return None    
    
    filtered_df = pd.merge(df, df_check, how='inner', on = ["participant_id", 'trial_id'])

        
    rows_removed = len(df)-len(filtered_df)
    
    print("Removed", rows_removed, "rows")
    if rows_removed > 0:
        print("Removed rows with [p_id, t_id]:\n", pd.unique(pd.concat([filtered_df,df]).drop_duplicates(keep=False)[["participant_id", "trial_id"]].values.ravel("K")))
    print()
    return filtered_df

def check_red_fixpoint(df: pd.DataFrame) -> pd.DataFrame:
    print("Checking if there are the correct amount of red fixpoints for the given experiment")
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
        return None
    filtered_df = pd.merge(df, df_check, how='inner', on = ["participant_id", 'trial_id'])

    
    rows_removed = len(df)-len(filtered_df)
    
    print("Removed", rows_removed, "rows")
    if rows_removed > 0:
        print("Removed rows with [p_id, t_id]:\n", pd.unique(pd.concat([filtered_df,df]).drop_duplicates(keep=False)[["participant_id", "trial_id"]].values.ravel("K")))
    print()
    return filtered_df

def check_white_fixpoint(df: pd.DataFrame) -> pd.DataFrame:
    print("Checking if there are the correct amount of white fixpoints for the given experiment")
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
        return None
    
    filtered_df = pd.merge(df, df_check, how='inner', on = ["participant_id", 'trial_id'])

    rows_removed = len(df)-len(filtered_df)
    
    print("Removed", rows_removed, "rows")
    if rows_removed > 0:
        print("Removed rows with [p_id, t_id]:\n", pd.unique(pd.concat([filtered_df,df]).drop_duplicates(keep=False)[["participant_id", "trial_id"]].values.ravel("K")))
    print()
    return filtered_df

def check_trial_var_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Checking if there are the correct amount of trial_var_data events for the given experiment")

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
    
    print("Removed", rows_removed, "rows")
    if rows_removed > 0:
        print("Removed rows with [p_id, t_id]:\n", pd.unique(pd.concat([filtered_df,df]).drop_duplicates(keep=False)[["participant_id", "trial_id"]].values.ravel("K")))
    print()
    return filtered_df

def check_start_event(df: pd.DataFrame) -> pd.DataFrame:
    print("Checking if there are the correct amount of start events for the given experiment")

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
 
    print("Removed", rows_removed, "rows")
    if rows_removed > 0:
        print("Removed rows with [p_id, t_id]:\n", pd.unique(pd.concat([filtered_df,df]).drop_duplicates(keep=False)[["participant_id", "trial_id"]].values.ravel("K")))
    print()
    return filtered_df

def check_end_event(df: pd.DataFrame) -> pd.DataFrame:
    print("Checking if there are the correct amount of end events for the given experiment")

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

    print("Removed", rows_removed, "rows")
    if rows_removed > 0:
        print("Removed rows with [p_id, t_id]:\n", pd.unique(pd.concat([filtered_df,df]).drop_duplicates(keep=False)[["participant_id", "trial_id"]].values.ravel("K")))
    print()
    return filtered_df


def clean_events(df:pd.DataFrame) -> pd.DataFrame:
    print("Starting event cleaning\n\n\n")
    cleaned_df = (df.
    pipe(exclude_nan_participants).
    pipe(exclude_special_participants, special_participants=SPECIAL_PARTICIPANTS).
    pipe(check_trialid_event).
    pipe(check_fixpoint_amount).
    pipe(check_red_fixpoint).
    pipe(check_white_fixpoint).
    pipe(check_trial_var_data).
    pipe(check_start_event).
    pipe(check_end_event)
    )
    return cleaned_df
    

def main():
    df = pd.read_parquet(RAW_DIR / "ANTI_SACCADE.pq")
    cleaned_df = clean_events(df)
    cleaned_df.to_parquet(CLEANED_DIR / "anti_saccade_processed.pq")
    

if __name__ == '__main__':
    main()





