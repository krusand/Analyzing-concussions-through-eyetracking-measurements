import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from functools import reduce
from tqdm import tqdm

def rename_columns(df: pd.DataFrame, prefix:str = '', suffix: str = '') -> pd.DataFrame:
    """Renames columns by joining multi-level column names with different delimiters. Used after aggregating data using dict aggregation method"""
    if prefix != '':
        prefix = prefix + '_'
    if suffix != '':
        suffix = '_' + suffix
    
    df.columns = [f"{col[0]}" if col[1] == '' else f"{prefix}{col[0]}_{col[1]}{suffix}" for col in df.columns.values]
    return df

def combine_samples_events(df_sample: pd.DataFrame, df_event: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """Combine sample data and event data to get fixpoints.
    
    Args:
        df_sample (pd.DataFrame): Dataframe with preprocessed sample data
        df_event (pd.DataFrame): Dataframe with preprocessed event data

    Returns:
        pd.DataFrame: Dataframe with fixpoints added to the sample data.
    """
    
    # Extract fixpoints
    df_fixpoints = df_event[df_event["event"]=="FIXPOINT"].loc[:,["participant_id", "trial_id", "time", "event", "colour", "stimulus_x", "stimulus_y"]]

    # Insert fixpoints in sample data
    df_sample = df_sample.copy()
    df_fixpoints = df_fixpoints.copy()

    # Make sure both DataFrames are sorted by time
    df_sample = df_sample.sort_values(["time", "trial_id", "participant_id"])
    df_fixpoints = df_fixpoints.sort_values(["time", "trial_id", "participant_id"])

    # Rename 'colour' column to 'fixpoint' so it's ready to merge
    df_fixpoints = df_fixpoints.rename(columns={"colour": "fixpoint"})

    # Perform a backward-looking join: for each row in sample_df, find the most recent fixpoint time
    if experiment == "FIXATIONS":
        df_sample = pd.merge_asof(
            df_sample,
            df_fixpoints,
            on="time",
            by=["participant_id", "trial_id"],
            direction="backward"
        )
    else:
        df_sample = pd.merge_asof(
            df_sample,
            df_fixpoints,
            on="time",
            by=["participant_id", "trial_id"],
            direction="nearest",
            tolerance=10
        )

    df_sample["fixpoint"] = df_sample["fixpoint"].map({RED:"red", GREEN:"green", BLUE:"blue", WHITE:"white"})
    
    return df_sample.sort_values(["participant_id", "trial_id", "time"])

###############
### GENERAL ###
###############

def get_pre_calculated_metrics_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns pd.Dataframe with columns ['experiment','participant_id', X_FEATURES],
    where X_FEATURES is a collection of features found by the following cartesian product:
    {'peak_velocity', 'amplitude', 'duration', 'avg_pupil_size'} x {np.mean, np.min, np.max, np.median, np.std}
    """
    

    features_sacc = (
        df
            .query("event == 'ESACC'")
            .groupby(["experiment","participant_id"])
            .agg({'peak_velocity': [np.mean, np.min, np.max, np.median, np.std],
                    'amplitude': [np.mean, np.min, np.max, np.median, np.std],
                    'duration': [np.mean, np.min, np.max, np.median, np.std]
    })
            .pipe(rename_columns, suffix='sacc')
    )
    features_fix = (
         df
            .query("event == 'EFIX'")
            .groupby(["experiment","participant_id"])
            .agg({'duration': [np.mean, np.min, np.max, np.median, np.std],
                    'avg_pupil_size': [np.mean, np.min, np.max, np.median, np.std]
    })
            .pipe(rename_columns, suffix='fix')
    )
    
    features_df = pd.merge(features_sacc, features_fix, how='outer', on = ["experiment", "participant_id"])
    
    return features_df

def get_disconjugacy_feature(df:pd.DataFrame) -> pd.DataFrame:
    logging.info("Extracting disconjugacy")
    
    if len(df.query("x_left.notnull() & x_right.notnull() & y_left.notnull() & y_right.notnull()"))==0:
        disconjugacy = (df
            .sort_values(["experiment", "participant_id", "trial_id", "time"])
            .groupby(["experiment", "participant_id"])
            .first()
            .assign(Var_total=None)
            .reset_index()
            [["experiment", "participant_id", "Var_total"]])
        return disconjugacy
    
    disconjugacy = (df
        .sort_values(["experiment", "participant_id", "trial_id", "time"])
        .query("x_left.notnull() & x_right.notnull() & y_left.notnull() & y_right.notnull()") # same as not null
        .groupby(["experiment", "participant_id"])
        .apply(lambda group: group.assign(
            x_left_rolling=group["x_left"].rolling(window=5, min_periods=1).mean(),
            x_right_rolling=group["x_right"].rolling(window=5, min_periods=1).mean(),
            y_left_rolling=group["y_left"].rolling(window=5, min_periods=1).mean(),
            y_right_rolling=group["y_right"].rolling(window=5, min_periods=1).mean()
        ))
        .reset_index(drop=True)
        .assign(
            X_diffs = lambda x: ((x["x_left_rolling"] - x["x_right_rolling"]) - 0)**2,
            Y_diffs = lambda x: ((x["y_left_rolling"] - x["y_right_rolling"]) - 0)**2
        )
        .groupby(["experiment", "participant_id"])
        .apply(lambda group: group.assign(
            X_squared_scaled = group["X_diffs"] / group.shape[0],
            Y_squared_scaled = group["Y_diffs"] / group.shape[0]
        ))
        .reset_index(drop=True)
        .groupby(["experiment", "participant_id"])
        .agg(
            Var_X = ("X_squared_scaled", "sum"),
            Var_Y = ("Y_squared_scaled", "sum")
        )
        .assign(
            Var_total = lambda x: x["Var_X"] + x["Var_Y"]
        )
        .reset_index()
        [["experiment", "participant_id", "Var_total"]]
    )
    return disconjugacy

def get_distance_between_fixations(df: pd.DataFrame) -> pd.DataFrame:
    """Finds acceleration features for anti saccade experiment

    Args:
        df (pd.DataFrame): Dataframe with preprocessed events

    Returns:
        pd.DataFrame: Dataframe with columns ['experiment','participant_id', X_FEATURES]
        where X_FEATURES is a collection of features found by the following cartesian product:
    """

    df = (df.query("event == 'EFIX'")
    .join((df
        .query("event == 'EFIX'")
        .groupby(["experiment", "participant_id", "trial_id", "eye"])[['x','y']].shift(1)
        .rename(columns={"x": "x_lagged", 
            "y": "y_lagged"})))
    .assign(
        x_fixation_dist = lambda x: x["x"] - x["x_lagged"],
        y_fixation_dist = lambda x: x["y"] - x["y_lagged"])
    .assign(
        fixation_distance = lambda x: np.sqrt( np.power(x["x_fixation_dist"],2) + np.power(x["y_fixation_dist"],2))
    )
    .groupby(["experiment", "participant_id"])
    .agg({'fixation_distance': [np.mean, np.min, np.max, np.median, np.std],
          'x_fixation_dist': [np.mean, np.min, np.max, np.median, np.std],
          'y_fixation_dist': [np.mean, np.min, np.max, np.median, np.std]
    })
    .reset_index()
    .pipe(rename_columns)
    )
    return df

def get_fixations_pr_second(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Adding fixations pr. second")
    df = (df.query("stimulus_active == True")
        .sort_values(by=["participant_id", "trial_id","time"])
        .assign(stimulus_time = lambda x: np.select([x.event == "FIXPOINT", x.event != "FIXPOINT"], [x.time, None]))
        .assign(stimulus_time = lambda x: x["stimulus_time"].ffill())
        .assign(max_event_time = lambda x: (
                x.sort_values(by=["participant_id", "trial_id", "time"])
                .groupby(["participant_id", "trial_id"])["time"]
                .transform(lambda group: (
                    group.iloc[-1]
                ))
            ))
        .assign(trial_active_duration_seconds = lambda x: (x["max_event_time"] - x["stimulus_time"])/1000)
        .query("event == 'EFIX'")
        .groupby(["experiment", "participant_id", "trial_id", "eye", "trial_active_duration_seconds"])
        .size()
        .reset_index(name="n_fixations")
        .assign(
            fixations_per_second_raw = lambda x: x["n_fixations"] / x["trial_active_duration_seconds"]
        )
        .groupby(["experiment", "participant_id", "trial_id"])
        .agg(
            total_fixations_per_second = ("fixations_per_second_raw", "sum"),
            n_eyes = ("eye", "nunique")
        )
        .reset_index()
        .assign(
            fixations_per_second = lambda x: x["total_fixations_per_second"] / x["n_eyes"]
        )
        .groupby(["experiment", "participant_id"])
        .agg({
            'fixations_per_second': [np.mean, np.min, np.max, np.median, np.std],
        })
        .reset_index()
        .pipe(rename_columns)
    )
    return df

def get_acceleration_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Finds acceleration features

    Args:
        df (pd.DataFrame): Dataframe with raw samples

    Returns:
        pd.DataFrame: Dataframe with columns ['experiment','participant_id', X_FEATURES]
        where X_FEATURES is a collection of features found by the following cartesian product:
        {'total_acceleration_magnitude', 'x_acceleration', 'y_acceleration'} x {np.mean, np.min, np.max, np.median, np.std}
    """
    logging.info("Extracting acceleration")
    acceleration = (df.join((df
    .groupby(["experiment", "participant_id", "trial_id"])[['x_velocity_left', 'y_velocity_left', 'x_velocity_right', 'y_velocity_right']].shift(1)
    .rename(columns={'x_velocity_left': 'x_velocity_left_lagged'
            , 'y_velocity_left': 'y_velocity_left_lagged'
            , 'x_velocity_right': 'x_velocity_right_lagged'
            , 'y_velocity_right': 'y_velocity_right_lagged'}))
    ).assign(x_acceleration_left = lambda x: (x["x_velocity_left"] - x["x_velocity_left_lagged"]) / (1/2000),
            y_acceleration_left = lambda x: (x["y_velocity_left"] - x["y_velocity_left_lagged"]) / (1/2000),
            x_acceleration_right = lambda x: (x["x_velocity_right"] - x["x_velocity_right_lagged"]) / (1/2000),
            y_acceleration_right = lambda x: (x["y_velocity_right"] - x["y_velocity_right_lagged"]) / (1/2000))
    .assign(x_acceleration = lambda x: np.nanmean(np.array(x["x_acceleration_left"], x["x_acceleration_right"])),
            y_acceleration = lambda x: np.nanmean(np.array(x["y_acceleration_left"], x["y_acceleration_right"])))
    .assign(total_acceleration_magnitude = lambda x: np.sqrt( np.power(x["x_acceleration"], 2) + np.power(x["y_acceleration"], 2)))
    .groupby(["experiment", "participant_id"])
    .agg({'total_acceleration_magnitude': [np.mean, np.min, np.max, np.median, np.std],
        'x_acceleration': [np.mean, np.min, np.max, np.median, np.std],
        'y_acceleration': [np.mean, np.min, np.max, np.median, np.std]
        })
    .reset_index()
    .pipe(rename_columns)
    )
    return acceleration


##################
## ANTI SACCADE ##
##################

def anti_saccade_get_trial_correctness_df(df:pd.DataFrame) -> pd.DataFrame:
    df_trials = (df
        .query('stimulus_active == True')
        .sort_values(by=["participant_id", "trial_id","time"])
        .assign(stimulus_time = lambda x: np.select([x.event == "FIXPOINT", x.event != "FIXPOINT"], [x.time, None]))
        .assign(stimulus_time = lambda x: x["stimulus_time"].ffill())
        .assign(saccade_direction = lambda x: np.select([(x["event"] == 'ESACC') & (np.abs(x["end_x"] - x["start_x"]) < 50),
                                                        (x["event"] == 'ESACC') & (x["end_x"] > x["start_x"]),
                                                        (x["event"] == 'ESACC') & (x["end_x"] < x["start_x"])],
                                                        ['no_direction',"right", "left"], default=None))
        .assign(saccade_end_area = lambda x: np.select([(x["event"] == 'ESACC') & ( 840 < x["end_x"]) & (x["end_x"] < 1080),
                                                        (x["event"] == 'ESACC') & (1080 <= x["end_x"]),
                                                        (x["event"] == 'ESACC') & (x["end_x"] <= 840)],
                                                    ['middle',"right", "left"], default=None))
        .assign(is_saccade_correct = lambda x: np.select([(x["saccade_direction"] == 'no_direction')
                                                        , (x["saccade_end_area"] == 'middle')
                                                        , (x["saccade_direction"] == x["saccade_end_area"]) & (x["saccade_direction"] != x["side"]) & (x["saccade_end_area"] != x["side"])
                                                        , (x["saccade_direction"] == x["saccade_end_area"]) & (x["saccade_direction"] == x["side"]) & (x["saccade_end_area"] == x["side"])
                                                        ],
                                                            [None, None, True, False], default=None)) 
        .assign(is_trial_correct = lambda x: (
                x.sort_values(by=["participant_id", "trial_id", "time"])
                .groupby(["participant_id", "trial_id"])["is_saccade_correct"]
                .transform(lambda group: (
                    True if not group.dropna().empty and group.dropna().iloc[0] == True else
                    False if not group.dropna().empty and group.dropna().iloc[0] == False else
                    None
                ))
            ))
    )
    return df_trials

def anti_saccade_get_n_correct_trials_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns pd.Dataframe with columns ['experiment', 'participant_id', 'n_correct_trials']
    """
    
    feature_df = (df
     .pipe(anti_saccade_get_trial_correctness_df)
     .groupby(["experiment","participant_id", "trial_id"])
     .agg(is_trial_correct = ('is_trial_correct', 'min')) 
     .reset_index()
     .groupby(["experiment", "participant_id"])
     .agg(n_correct_trials = ('is_trial_correct', 'sum'))
     .reset_index()
    [["experiment", "participant_id", "n_correct_trials"]]
    )
    
    return feature_df

def anti_saccade_get_prop_trials_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns pd.Dataframe with columns ['experiment', 'participant_id', 'prop_correct_trials']
    """
    
    feature_df = (df
     .pipe(anti_saccade_get_trial_correctness_df)
     .groupby(["experiment","participant_id", "trial_id"])
     .agg(is_trial_correct = ('is_trial_correct', 'min')) 
     .reset_index()
     .groupby(["experiment", "participant_id"])
     .agg(n_correct_trials = ('is_trial_correct', 'sum'),
          n_trials = ('is_trial_correct', 'count'))
     .reset_index()
     .assign(prop_correct_trials = lambda x: x["n_correct_trials"] / x["n_trials"])
     [["experiment", "participant_id", "prop_correct_trials"]]
    )
    
    return feature_df

def anti_saccade_get_reaction_time_feature(df: pd.DataFrame) -> pd.DataFrame:
    return (df
        .query('stimulus_active == True')
        .pipe(anti_saccade_get_trial_correctness_df)
        .sort_values(by=["participant_id", "trial_id", "time"])
        .assign(is_saccade_correct = lambda x: np.select([ (x["is_saccade_correct"] == True) ], [True], default=None))
        .query("is_saccade_correct == True")
        .groupby(["experiment","participant_id", "trial_id"])
        .first()
        .reset_index()
        .assign(reaction_time = lambda group: group["start_time"] - group["stimulus_time"])
        .groupby(["experiment", "participant_id"])
        .agg(reaction_time_avg = ('reaction_time', 'mean'),
             reaction_time_std = ('reaction_time', 'std'))
        .reset_index()
    )
    
def get_anti_saccade_features(event_features:bool, sample_features:bool) -> pd.DataFrame:
    """Runs all anti saccade features extractions

    Args:
        df_event (pd.DataFrame): The preprocessed event dataframe
        df_samples (pd.DataFrame): The preprocessed sample dataframe

    Returns:
        pd.DataFrame: Dataframe with columns ["experiment", "participant_id", X_FEATURES], where X_FEATURES is a collection of features
    """

    logging.info("Extracting anti saccade features")

    experiment = "ANTI_SACCADE"     
    df_event_features_list=[]
    df_sample_features_list=[]
    
    if event_features:
        logging.info("Starting event feature extraction")
        df_event = pd.read_parquet(PREPROCESSED_DIR / f"{experiment}_events.pq")

        event_feature_functions = [get_pre_calculated_metrics_feature, anti_saccade_get_n_correct_trials_feature, anti_saccade_get_prop_trials_feature, anti_saccade_get_reaction_time_feature]
        df_event_features_list = [f(df=df_event) for f in event_feature_functions]

    if sample_features:
        logging.info("Starting sample feature extraction")
        df_sample = (pd.read_parquet(PREPROCESSED_DIR / f'{experiment}_samples.pq')
        .sort_values(["experiment", "participant_id", "trial_id","time"])
        )
        sample_feature_functions = [get_acceleration_feature, get_disconjugacy_feature]
        df_sample_features_list = [f(df=df_sample) for f in sample_feature_functions]
    
    df_features_list = df_event_features_list + df_sample_features_list
    
    df_features = reduce(lambda x, y: pd.merge(x, y, on = ["experiment", "participant_id"]), df_features_list)
    
    logging.info("Finished extracting anti saccade features")
    
    return df_features

#########################
## REACTION/PROSACCADE ##
#########################


def reaction_add_saccade_direction(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Adding saccade direction")
    df = (df
        .assign(saccade_direction_x = lambda x: np.select([(x["event"] == 'ESACC') & (np.abs(x["end_x"] - x["start_x"]) < 50),
                                                        (x["event"] == 'ESACC') & (x["end_x"] > x["start_x"]),
                                                        (x["event"] == 'ESACC') & (x["end_x"] < x["start_x"])],
                                                        ['no_direction',"right", "left"], default=None))
        .assign(saccade_direction_y = lambda x: np.select([(x["event"] == 'ESACC') & (np.abs(x["end_y"] - x["start_y"]) < 50),
                                                        (x["event"] == 'ESACC') & (x["end_y"] > x["start_y"]),
                                                        (x["event"] == 'ESACC') & (x["end_y"] < x["start_y"])],
                                                        ['no_direction',"up", "down"], default=None))
        .assign(saccade_direction = lambda x: np.select([(x["saccade_direction_x"].isin(["right", "left"])) & (x["saccade_direction_y"].isin(["up", "down"]))
                                                         , (x["saccade_direction_x"].isin(["right", "left"])) & (x["saccade_direction_y"] == "no_direction")
                                                         , (x["saccade_direction_x"] == "no_direction") & (x["saccade_direction_y"].isin(["down", "up"]))
                                                         , (x["saccade_direction_x"] == "no_direction") & (x["saccade_direction_y"] == "no_direction")
                                                        ]
                                                        , [(x["saccade_direction_x"] + "_" + x["saccade_direction_y"])
                                                         , (x["saccade_direction_x"])
                                                         , (x["saccade_direction_y"])
                                                         , 'no_direction'
                                                        ]
                                                        , default=None))
    )
    return df

def reaction_add_stimulus_direction(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Adding stimulus direction")

    MIDDLE_FIXPOINT_X=960
    MIDDLE_FIXPOINT_Y=540
    
    df = (df
        .assign(stimulus_direction_x = lambda x: np.select([(x["event"] == 'FIXPOINT') & (np.abs(x["stimulus_x"] - MIDDLE_FIXPOINT_X) < 50),
                                                        (x["event"] == 'FIXPOINT') & (x["stimulus_x"] > MIDDLE_FIXPOINT_X),
                                                        (x["event"] == 'FIXPOINT') & (x["stimulus_x"] < MIDDLE_FIXPOINT_X)],
                                                        ["middle", "right", "left"], default=None))
        .assign(stimulus_direction_y = lambda x: np.select([(x["event"] == 'FIXPOINT') & (np.abs(x["stimulus_y"] - MIDDLE_FIXPOINT_Y) < 50),
                                                        (x["event"] == 'FIXPOINT') & (x["stimulus_y"] > MIDDLE_FIXPOINT_Y),
                                                        (x["event"] == 'FIXPOINT') & (x["stimulus_y"] < MIDDLE_FIXPOINT_Y)],
                                                        ['middle',"up", "down"], default=None))
        .assign(stimulus_direction = lambda x: np.select([(x["stimulus_direction_x"].isin(["right", "left"])) & (x["stimulus_direction_y"].isin(["up", "down"]))
                                                         , (x["stimulus_direction_x"].isin(["right", "left"])) & (x["stimulus_direction_y"] == "middle")
                                                         , (x["stimulus_direction_x"] == "middle") & (x["stimulus_direction_y"].isin(["right", "left"]))
                                                         , (x["stimulus_direction_x"] == "middle") & (x["stimulus_direction_y"] == "middle")
                                                        ]
                                                        , [(x["stimulus_direction_x"] + "_" + x["stimulus_direction_y"])
                                                         , (x["stimulus_direction_x"])
                                                         , (x["stimulus_direction_x"])
                                                         , 'middle'
                                                        ]
                                                        , default=None)).ffill()
    )
    return df

def reaction_get_trial_correctness_df(df:pd.DataFrame) -> pd.DataFrame:
    logging.info("Extracting trial correctness")

    df_trials = (df
            .query('stimulus_active == True')
            .sort_values(by=["participant_id", "trial_id","time"])
            .assign(stimulus_time = lambda x: np.select([x.event == "FIXPOINT", x.event != "FIXPOINT"], [x.time, None]))
            .assign(stimulus_time = lambda x: x["stimulus_time"].ffill())
            .pipe(reaction_add_stimulus_direction)
            .pipe(reaction_add_saccade_direction)
            .assign(is_saccade_correct = lambda x: np.select([ (x["saccade_direction"].notna()) & (x["saccade_direction"] == 'no_direction')
                                                            , (x["saccade_direction"].notna()) & (x["saccade_direction"] == x["stimulus_direction"])
                                                            , (x["saccade_direction"].notna()) & (x["saccade_direction"] != x["stimulus_direction"])
                                                            ],
                                                            [False, True, False], default=None)) 
            .assign(is_trial_correct = lambda x: (
                    x.sort_values(by=["participant_id", "trial_id", "time"])
                    .groupby(["participant_id", "trial_id"])["is_saccade_correct"]
                    .transform(lambda group: (
                        True if not group.dropna().empty and group.dropna().iloc[0] == True else
                        False if not group.dropna().empty and group.dropna().iloc[0] == False else
                        None
                    ))
                ))
    )
    return df_trials

def reaction_get_n_correct_trials_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns pd.Dataframe with columns ['experiment', 'participant_id', 'n_correct_trials']
    """
    logging.info("Extracting n correct trials")

    feature_df = (df
     .pipe(reaction_get_trial_correctness_df)
     .groupby(["experiment","participant_id", "trial_id"])
     .agg(is_trial_correct = ('is_trial_correct', 'min')) 
     .reset_index()
     .groupby(["experiment", "participant_id"])
     .agg(n_correct_trials = ('is_trial_correct', 'sum'))
     .reset_index()
    [["experiment", "participant_id", "n_correct_trials"]]
    )
    
    return feature_df

def reaction_get_prop_trials_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns pd.Dataframe with columns ['experiment', 'participant_id', 'prop_correct_trials']
    """
    logging.info("Extracting prop correct trials")

    feature_df = (df
     .pipe(reaction_get_trial_correctness_df)
     .groupby(["experiment","participant_id", "trial_id"])
     .agg(is_trial_correct = ('is_trial_correct', 'min')) 
     .reset_index()
     .groupby(["experiment", "participant_id"])
     .agg(n_correct_trials = ('is_trial_correct', 'sum'),
          n_trials = ('is_trial_correct', 'count'))
     .reset_index()
     .assign(prop_correct_trials = lambda x: x["n_correct_trials"] / x["n_trials"])
     [["experiment", "participant_id", "prop_correct_trials"]]
    )
    
    return feature_df

def reaction_get_reaction_time_feature(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Extracting reaction time")

    return (df
        .query('stimulus_active == True')
        .pipe(reaction_get_trial_correctness_df)
        .sort_values(by=["participant_id", "trial_id", "time"])
        .assign(is_saccade_correct = lambda x: np.select([ (x["is_saccade_correct"] == True) ], [True], default=None))
        .query("is_saccade_correct == True")
        .groupby(["experiment","participant_id", "trial_id"])
        .first()
        .reset_index()
        .assign(reaction_time = lambda group: group["start_time"] - group["stimulus_time"])
        .groupby(["experiment", "participant_id"])
        .agg(reaction_time_avg = ('reaction_time', 'mean'),
             reaction_time_std = ('reaction_time', 'std'))
        .reset_index()
    )

###############
## FITTS LAW ##
###############

def fitts_law_add_nearest_fixation_overshoot(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Adding fixation overshoot with piping")

    df = (
        df.sort_values(["participant_id", "trial_id", "time"])
          .groupby(["participant_id", "trial_id"], group_keys=False)
          .apply(lambda group: (
              group.assign(
                  distance_to_stimulus_1 = lambda x: np.sqrt(
                      np.power(x["x"] - group.query("event == 'FIXPOINT'").iloc[0]["stimulus_x"], 2) +
                      np.power(x["y"] - group.query("event == 'FIXPOINT'").iloc[0]["stimulus_y"], 2)
                  ),
                  distance_to_stimulus_2 = lambda x: np.sqrt(
                      np.power(x["x"] - group.query("event == 'FIXPOINT'").iloc[1]["stimulus_x"], 2) +
                      np.power(x["y"] - group.query("event == 'FIXPOINT'").iloc[1]["stimulus_y"], 2)
                  )
              )
          ))
          .assign(
              distance_to_nearest_stimulus = lambda x: x[["distance_to_stimulus_1", "distance_to_stimulus_2"]].min(axis=1)
          )
    )
    return df

def fitts_law_get_fixation_overshoot(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Calculating fixation overshoot summary with eye scaling")

    df = (df
        .query("stimulus_active == True")
        .sort_values(by=["participant_id", "trial_id", "time"])
        .assign(
            stimulus_time = lambda x: np.select(
                [x.event == "FIXPOINT", x.event != "FIXPOINT"],
                [x.time, None]
            )
        )
        .assign(stimulus_time = lambda x: x["stimulus_time"].ffill())
        .pipe(fitts_law_add_nearest_fixation_overshoot)
        .query("event == 'EFIX'")
        .groupby(["experiment", "participant_id", "trial_id", "eye"])
        .agg(
            avg_fixation_overshoot_eye=('distance_to_nearest_stimulus', 'mean')
        )
        .reset_index()
        .groupby(["experiment", "participant_id", "trial_id"])
        .agg(
            total_fixation_overshoot=('avg_fixation_overshoot_eye', 'sum'),
            n_eyes=('eye', 'nunique')
        )
        .reset_index()
        .assign(
            fixation_overshoot = lambda x: x["total_fixation_overshoot"] / x["n_eyes"]
        )
        .groupby(["experiment", "participant_id"])
        .agg({
            'fixation_overshoot': [np.mean, np.min, np.max, np.median, np.std]
        })
        .reset_index()
        .pipe(rename_columns)
    )

    return df

#################
## KING DEVICK ##
#################

def king_devick_get_avg_mistakes_pr_trial(df: pd.DataFrame) -> pd.DataFrame:
    df = (df
          .query("event == 'TRIAL_VAR_DATA'")
          .groupby(["experiment", "participant_id"])
          .agg(avg_mistakes_pr_trial = ('marks', 'mean'))
          .reset_index()
    )
    return df
    
def king_devick_get_avg_time_elapsed_pr_trial(df: pd.DataFrame) -> pd.DataFrame:
    df = (df
          .query("event == 'TRIAL_VAR_DATA'")
          .groupby(["experiment", "participant_id"])
          .agg(avg_time_elapsed_pr_trial = ('time_elapsed', 'mean'))
          .reset_index()
    )
    return df

def king_devick_get_pct_wrong_directional_saccades(df: pd.DataFrame) -> pd.DataFrame:

    
    logging.info("Adding saccades pr. second")
    df = (df
        .sort_values(by=["participant_id", "trial_id","time"])
        .assign(saccade_direction = lambda x: np.select([
                                                        (x["event"] == 'ESACC') & (x["end_x"] > x["start_x"]),
                                                        (x["event"] == 'ESACC') & (x["end_x"] < x["start_x"])],
                                                        ["right", "left"], default=None))
        .query("event == 'ESACC'")
        .query("time/1000 < time_elapsed") # acts as stimulus_active
        .groupby(["experiment", "participant_id", "trial_id", "eye", "saccade_direction"])
        .size()
        .reset_index(name="n_saccades")
        .assign(n_trial_saccades = lambda x: x.groupby(["participant_id", "trial_id", "eye"])["n_saccades"].transform('sum'))
        .assign(pct_saccade = lambda x: 100* x["n_saccades"] / x["n_trial_saccades"])
        .query("saccade_direction == 'left'")
        .groupby(["experiment", "participant_id", "trial_id"])
        .agg(
            total_wrong_saccade_pct=('pct_saccade', 'sum'),
            n_eyes=('eye', 'nunique')
        )
        .reset_index()
        .assign(
            wrong_direction_saccade_pct = lambda x: x["total_wrong_saccade_pct"] / x["n_eyes"]
        )
        .groupby(["experiment", "participant_id"])
        .agg({
            'wrong_direction_saccade_pct' : [np.mean, np.min, np.max, np.median, np.std]
        })
        .reset_index()
        .pipe(rename_columns)
    )
    return df

def king_devick_get_saccades_pr_second(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Calculating saccades per second with eye scaling")

    df = (df
        .sort_values(by=["participant_id", "trial_id", "time"])
        .assign(
            trial_active_duration_seconds = lambda x: (
                x.groupby(["participant_id", "trial_id"])["time_elapsed"]
                 .transform('first')
            )
        )
        .query("event == 'ESACC'")
        .query("time / 1000 < time_elapsed")
        .groupby(["experiment", "participant_id", "trial_id", "eye", "trial_active_duration_seconds"])
        .size()
        .reset_index(name="n_saccades")
        .assign(
            saccades_per_second = lambda x: x["n_saccades"] / x["trial_active_duration_seconds"]
        )
        .groupby(["experiment", "participant_id", "trial_id"])
        .agg(
            total_saccades_per_second=('saccades_per_second', 'sum'),
            n_eyes=('eye', 'nunique')
        )
        .reset_index()
        .assign(
            saccades_per_second = lambda x: x["total_saccades_per_second"] / x["n_eyes"]
        )
        .groupby(["experiment", "participant_id"])
        .agg({
            'saccades_per_second':  [np.mean, np.min, np.max, np.median, np.std]
        })
        .reset_index()
        .pipe(rename_columns)
    )

    return df


##################
## EVIL BASTARD ##
##################

def get_distance_to_stimulus_features(df: pd.DataFrame) -> pd.DataFrame:
    features = (df
        .assign(
            distance_to_fixpoint_left_x = lambda x: np.power(x["x_left"]-x["stimulus_x"], 2),
            distance_to_fixpoint_left_y = lambda x: np.power(x["y_left"]-x["stimulus_y"], 2),
            distance_to_fixpoint_right_x = lambda x: np.power(x["x_right"]-x["stimulus_x"], 2),
            distance_to_fixpoint_right_y = lambda x: np.power(x["y_right"]-x["stimulus_y"], 2),
            distance_to_fixpoint_left = lambda x: np.sqrt(x["distance_to_fixpoint_left_x"] + x["distance_to_fixpoint_left_y"]),
            distance_to_fixpoint_right = lambda x: np.sqrt(x["distance_to_fixpoint_right_x"] + x["distance_to_fixpoint_right_y"])
        )
        .assign(
            distance_to_fixpoint_x = lambda x: np.select(
                [( x["distance_to_fixpoint_left_x"].notna() & x["distance_to_fixpoint_right_x"].notna() )
                    ,( x["distance_to_fixpoint_left_x"].notna() )
                    ,( x["distance_to_fixpoint_right_x"].notna() ) 
                ]
                , [  (x["distance_to_fixpoint_left_x"] + x["distance_to_fixpoint_right_x"]) / 2
                    , x["distance_to_fixpoint_left_x"]
                    , x["distance_to_fixpoint_right_x"]
                ], default=None),
            distance_to_fixpoint_y = lambda x: np.select(
                [( x["distance_to_fixpoint_left_y"].notna() & x["distance_to_fixpoint_right_y"].notna() )
                    ,( x["distance_to_fixpoint_left_y"].notna() )
                    ,( x["distance_to_fixpoint_right_y"].notna() ) 
                ]
                , [  (x["distance_to_fixpoint_left_y"] + x["distance_to_fixpoint_right_y"]) / 2
                    , x["distance_to_fixpoint_left_y"]
                    , x["distance_to_fixpoint_right_y"]
                ], default=None),
            distance_to_fixpoint = lambda x: np.select(
                [ ( x["distance_to_fixpoint_left"].notna() & x["distance_to_fixpoint_right"].notna() )
                 , x["distance_to_fixpoint_left"].notna()
                 , x["distance_to_fixpoint_right"].notna()
                    
                ]
                , [ (x["distance_to_fixpoint_left"] + x["distance_to_fixpoint_right"]) / 2
                   , x['distance_to_fixpoint_left']
                   , x["distance_to_fixpoint_right"]   
                ]
                , 
                default=None
            )
        )
        .groupby(["experiment", "participant_id"])
        .agg({
            'distance_to_fixpoint': ["mean", "min", "max", "median", "std"],
            'distance_to_fixpoint_x': ["mean", "min", "max", "median", "std"],
            'distance_to_fixpoint_y': ["mean", "min", "max", "median", "std"],
        })
        .reset_index()
        .pipe(rename_columns)
    )
    
    return features


############
## SHAPES ##
############

####################
## SMOOTH PURSUIT ##
####################

###################
###   General   ###
###################

EXPERIMENT_EVENT_FEATURE_MAP = {
    # DEBUGGING:
    
    "ANTI_SACCADE" : [get_pre_calculated_metrics_feature, anti_saccade_get_n_correct_trials_feature, anti_saccade_get_prop_trials_feature, anti_saccade_get_reaction_time_feature],
    "REACTION" : [get_pre_calculated_metrics_feature, reaction_get_n_correct_trials_feature, reaction_get_prop_trials_feature, reaction_get_reaction_time_feature],
    "FITTS_LAW" : [get_pre_calculated_metrics_feature, fitts_law_get_fixation_overshoot, get_fixations_pr_second],
    "KING_DEVICK" : [get_pre_calculated_metrics_feature, king_devick_get_avg_mistakes_pr_trial, king_devick_get_avg_time_elapsed_pr_trial, king_devick_get_pct_wrong_directional_saccades, king_devick_get_saccades_pr_second],
    "EVIL_BASTARD" : [get_pre_calculated_metrics_feature, get_distance_between_fixations, get_fixations_pr_second],
    "SHAPES" : [get_pre_calculated_metrics_feature, get_distance_between_fixations, get_fixations_pr_second],
    "SMOOTH_PURSUITS" : [get_pre_calculated_metrics_feature, get_distance_between_fixations, get_fixations_pr_second]
}
EXPERIMENT_SAMPLE_FEATURE_MAP = {
    "ANTI_SACCADE" : [get_acceleration_feature, get_disconjugacy_feature],
    "REACTION" : [get_acceleration_feature, get_disconjugacy_feature],
    "FITTS_LAW" : [get_acceleration_feature, get_disconjugacy_feature],
    "KING_DEVICK" : [get_acceleration_feature, get_disconjugacy_feature],
    "EVIL_BASTARD" : [get_acceleration_feature, get_disconjugacy_feature],
    "SHAPES" : [get_acceleration_feature, get_disconjugacy_feature],
    "SMOOTH_PURSUITS" : [get_acceleration_feature, get_disconjugacy_feature],
    "FIXATION" : [get_acceleration_feature, get_disconjugacy_feature]
}
EXPERIMENT_COMBINED_FEATURE_MAP = {
    "EVIL_BASTARD" : [get_distance_to_stimulus_features],
    "SHAPES" : [get_distance_to_stimulus_features],
    "SMOOTH_PURSUITS" : [get_distance_to_stimulus_features],
    "FIXATIONS": [get_distance_to_stimulus_features]
}

def get_features(experiment: str, event_features: bool, sample_features: bool) -> pd.DataFrame:
    """Runs all feature extractions

    Returns:
        pd.DataFrame: Dataframe with columns ["experiment", "participant_id", X_FEATURES], where X_FEATURES is a collection of features
    """

    logging.info(f"Extracting {experiment} features")
    
    # Read participant and trial id to identify unique participants
    df_index = pd.read_parquet(
        f"{PREPROCESSED_DIR}/{experiment}_events.pq", 
        columns=["participant_id"]
    )
    participant_groups = df_index["participant_id"].unique()
    
    df_features_all_participants = []
    df_event_features_list = []
    df_sample_features_list = []
    df_combined_features_list = []
    
    for participant_id in tqdm(participant_groups, total=len(participant_groups)):
        logging.info(f"Processing participant {participant_id}")

        filters = [('participant_id', '=', participant_id)]
        
        if experiment in EXPERIMENT_EVENT_FEATURE_MAP and event_features:
            logging.info("Starting event feature extraction")
            df_event = pd.read_parquet(PREPROCESSED_DIR / f"{experiment}_events.pq", filters=filters)
            event_feature_functions = EXPERIMENT_EVENT_FEATURE_MAP[experiment]
            df_event_features_list = [f(df=df_event) for f in event_feature_functions]

        if experiment in EXPERIMENT_SAMPLE_FEATURE_MAP and sample_features:
            logging.info("Starting sample feature extraction")
            df_sample = (pd.read_parquet(PREPROCESSED_DIR / f'{experiment}_samples.pq', filters=filters)
            .sort_values(["experiment", "participant_id", "trial_id","time"])
            )
            sample_feature_functions = EXPERIMENT_SAMPLE_FEATURE_MAP[experiment]
            df_sample_features_list = [f(df=df_sample) for f in sample_feature_functions]
        
        if experiment in EXPERIMENT_COMBINED_FEATURE_MAP and (event_features and sample_features):
            logging.info("Starting combined feature extraction")
            df_combined = combine_samples_events(df_sample, df_event, experiment)        
            combined_feature_functions = EXPERIMENT_COMBINED_FEATURE_MAP[experiment]
            df_combined_features_list = [f(df=df_combined) for f in combined_feature_functions]
        
        df_features_par_list = df_event_features_list + df_sample_features_list + df_combined_features_list
    
        df_features_par = reduce(lambda x, y: pd.merge(x, y, on = ["experiment", "participant_id"]), df_features_par_list)

        df_features_all_participants.append(df_features_par)
    
    df_features = pd.concat(df_features_all_participants, ignore_index=True)
    
    logging.info(f"Finished extracting {experiment} features")
    
    return df_features





