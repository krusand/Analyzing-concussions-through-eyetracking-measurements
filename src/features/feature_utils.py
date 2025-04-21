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

def rename_columns(df):
    """Renames columns by joining multi-level column names with different delimiters."""
    # Iterate over all column names
    df.columns = [f"{col[0]}" if col[1] == '' else f"{col[0]}_{col[1]}" for col in df.columns.values]
    return df

def combine_samples_events(df_sample: pd.DataFrame, df_event: pd.DataFrame) -> pd.DataFrame:
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
    df_sample = pd.merge_asof(
        df_sample,
        df_fixpoints,
        on="time",
        by=["participant_id", "trial_id"],
        direction="nearest",
        tolerance=10
    )

    df_sample["fixpoint"] = df_sample["fixpoint"].map({RED:"red", GREEN:"green", BLUE:"blue", WHITE:"white"})
    
    return df_sample

###############
### GENERAL ###
###############

def get_pre_calculated_metrics_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns pd.Dataframe with columns ['experiment','participant_id', X_FEATURES],
    where X_FEATURES is a collection of features found by the following cartesian product:
    {'peak_velocity', 'amplitude', 'duration', 'avg_pupil_size'} x {np.mean, np.min, np.max, np.median, np.std}
    """
    features_df = (df.groupby(["experiment", "participant_id"])
    .agg(
        mean_peak_velocity_sacc = ('peak_velocity', lambda x: x[df.loc[x.index, 'event'] == 'ESACC'].mean()),
        mean_amplitude_sacc = ('amplitude', lambda x: x[df.loc[x.index, 'event'] == 'ESACC'].mean()),
        mean_duration_sacc = ('duration', lambda x: x[df.loc[x.index, 'event'] == 'ESACC'].mean()),
        mean_duration_fix = ('duration', lambda x: x[df.loc[x.index, 'event'] == 'EFIX'].mean()),
        mean_pupil_size_fix = ('avg_pupil_size', lambda x: x[df.loc[x.index, 'event'] == 'EFIX'].mean()),
    )
    .reset_index()
    )    
    return features_df

def get_acceleration_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Finds acceleration features for anti saccade experiment

    Args:
        df (pd.DataFrame): Dataframe with raw samples

    Returns:
        pd.DataFrame: Dataframe with columns ['experiment','participant_id', X_FEATURES]
        where X_FEATURES is a collection of features found by the following cartesian product:
        {'total_acceleration_magnitude_left', 'total_acceleration_magnitude_right'} x {np.mean, np.min, np.max, np.median, np.std}
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
    .assign(total_acceleration_magnitude_left = lambda x: np.sqrt( np.power(x["x_acceleration_left"], 2) + np.power(x["y_acceleration_left"], 2)),
            total_acceleration_magnitude_right = lambda x: np.sqrt( np.power(x["x_acceleration_right"], 2) + np.power(x["y_acceleration_right"], 2)))
    .groupby(["experiment", "participant_id"])
    .agg({'total_acceleration_magnitude_left': [np.mean, np.min, np.max, np.median, np.std],
        'total_acceleration_magnitude_right': [np.mean, np.min, np.max, np.median, np.std]
        })
    .reset_index()
    .pipe(rename_columns)
    )
    return acceleration

# Eye disconjugacy
# Paper: https://www.liebertpub.com/doi/full/10.1089/neu.2014.3687

def get_disconjugacy_feature(df:pd.DataFrame) -> pd.DataFrame:
    logging.info("Extracting disconjugacy")
    
    if len(df.query("x_left == x_left & x_right == x_right & y_left == y_left & y_right == y_right"))==0:
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
        .query("x_left == x_left & x_right == x_right & y_left == y_left & y_right == y_right") # same as not null
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
    .agg({'fixation_distance': [np.mean, np.std],
    })
    .reset_index()
    .pipe(rename_columns)
    )
    return df

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

def get_reaction_features(event_features:bool, sample_features:bool) -> pd.DataFrame:
    """Runs all reaction feature extractions

    Args:
        df_event (pd.DataFrame): The preprocessed event dataframe
        df_samples (pd.DataFrame): The preprocessed sample dataframe
        
    Returns:
        pd.DataFrame: Dataframe with columns ["experiment", "participant_id", X_FEATURES], where X_FEATURES is a collection of features
    """
    logging.info("Extracting reaction features")
    
    experiment = "REACTION"
    df_event_features_list=[]
    df_sample_features_list=[]
    

    if event_features:
        logging.info("Starting event feature extraction")
        df_event = pd.read_parquet(PREPROCESSED_DIR / f"{experiment}_events.pq")

        event_feature_functions = [get_pre_calculated_metrics_feature, reaction_get_n_correct_trials_feature, reaction_get_prop_trials_feature, reaction_get_reaction_time_feature]
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
   
    logging.info("Finished extracting reaction features")

    return df_features
    

###############
## FITTS LAW ##
###############

def fitts_law_add_saccade_direction(df: pd.DataFrame) -> pd.DataFrame:
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

def fitts_law_add_stimulus_direction(df: pd.DataFrame) -> pd.DataFrame:
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
                                                        , default=None))
    )
    return df

def fitts_law_add_fixation_area(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Adding fixation area")
    
    MIDDLE_FIXPOINT_X=960
    MIDDLE_FIXPOINT_Y=540
    
    df = (df
        .assign(fixation_area_x = lambda x: np.select([(x["event"] == 'EFIX') & (np.abs(x["x"] - MIDDLE_FIXPOINT_X) < 50),
                                                        (x["event"] == 'EFIX') & (x["x"] > MIDDLE_FIXPOINT_X),
                                                        (x["event"] == 'EFIX') & (x["x"] < MIDDLE_FIXPOINT_X)],
                                                        ["middle", "right", "left"], default=None))
        .assign(fixation_area_y = lambda x: np.select([(x["event"] == 'EFIX') & (np.abs(x["y"] - MIDDLE_FIXPOINT_Y) < 50),
                                                        (x["event"] == 'EFIX') & (x["y"] > MIDDLE_FIXPOINT_Y),
                                                        (x["event"] == 'EFIX') & (x["y"] < MIDDLE_FIXPOINT_Y)],
                                                        ["middle", "up", "down"], default=None))
        .assign(fixation_area = lambda x: np.select([(x["fixation_area_x"].isin(["right", "left"])) & (x["fixation_area_y"].isin(["up", "down"]))
                                                         , (x["fixation_area_x"].isin(["right", "left"])) & (x["fixation_area_y"] == "middle")
                                                         , (x["fixation_area_x"] == "middle") & (x["fixation_area_y"].isin(["right", "left"]))
                                                         , (x["fixation_area_x"] == "middle") & (x["fixation_area_y"] == "middle")
                                                        ]
                                                        , [(x["fixation_area_x"] + "_" + x["fixation_area_y"])
                                                         , (x["fixation_area_x"])
                                                         , (x["fixation_area_x"])
                                                         , 'middle'
                                                        ]
                                                        , default=None))
    )
    return df

def fitts_law_get_fixations_pr_second(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Adding fixations pr. second")
    df = (df.query("stimulus_active == True")
        .query("(eye == 'R') or (eye != eye)") # right eye or is na
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
        .groupby(["experiment", "participant_id", "trial_id", "trial_active_duration_seconds"])
        .size()
        .reset_index(name="n_fixations")
        .assign(fixations_per_second=lambda x: x["n_fixations"] / x["trial_active_duration_seconds"])
        .groupby(["experiment", "participant_id"])
        .agg(avg_fixations_pr_second = ('fixations_per_second', 'mean'),
             std_fixations_pr_second = ('fixations_per_second', 'std'))
        .reset_index()
    )
    return df

def fitts_law_add_nearest_fixation_overshoot(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Adding fixation overshoot")
    result = []
    
    grouped_df = df.sort_values(['participant_id', 'trial_id', 'time']).groupby(['participant_id', 'trial_id'])
    
    for (participant, trial), group in tqdm(grouped_df):
        group = group.copy()
        
        stimulus = group.query("event == 'FIXPOINT'")
        
        stimulus_coords = [
            (stimulus.iloc[0]["stimulus_x"], stimulus.iloc[0]["stimulus_y"]),
            (stimulus.iloc[1]["stimulus_x"], stimulus.iloc[1]["stimulus_y"])
        ]
        
        def compute_distances(row):
            for i, (sx, sy) in enumerate(stimulus_coords, start=1):
                row[f"distance_to_stimulus_{i}"] = np.sqrt(np.power((row["x"] - sx),2) + np.power((row["y"] - sy),2))
            return row
        
        group = group.apply(compute_distances, axis=1)
        result.append(group)
        
    df_with_distances = pd.concat(result, ignore_index=True)

    df_with_distances['distance_to_nearest_stimulus'] = df_with_distances[['distance_to_stimulus_1','distance_to_stimulus_2']].min(axis=1)

    return df_with_distances

def fitts_law_get_fixation_overshoot(df: pd.DataFrame) -> pd.DataFrame:
            
    df = (df
    .query("stimulus_active == True")
    .query("(eye == 'R') or (eye != eye)") # right eye or is na
    .sort_values(by=["participant_id", "trial_id","time"])
    .assign(stimulus_time = lambda x: np.select([x.event == "FIXPOINT", x.event != "FIXPOINT"], [x.time, None]))
    .assign(stimulus_time = lambda x: x["stimulus_time"].ffill())
    .pipe(fitts_law_add_saccade_direction)
    .pipe(fitts_law_add_stimulus_direction)
    .pipe(fitts_law_add_fixation_area) 
    .pipe(fitts_law_add_nearest_fixation_overshoot)
    .groupby(["experiment", "participant_id"])
    .agg(avg_fixation_overshoot = ('distance_to_nearest_stimulus', 'mean'),
         std_fixation_overshoot = ('distance_to_nearest_stimulus', 'std'))
    .reset_index()
    )

    
    return df

def get_fitts_law_features(event_features:bool, sample_features:bool) -> pd.DataFrame:
    """Runs all fitts law features extractions

    Args:
        df_event (pd.DataFrame): The preprocessed event dataframe
        df_samples (pd.DataFrame): The preprocessed sample dataframe

    Returns:
        pd.DataFrame: Dataframe with columns ["experiment", "participant_id", X_FEATURES], where X_FEATURES is a collection of features
    """
    logging.info("Starting fitts law feature extraction")
    
    experiment = "FITTS_LAW"
    df_event_features_list=[]
    df_sample_features_list=[]
    

    if event_features:
        logging.info("Starting event feature extraction")
        df_event = pd.read_parquet(PREPROCESSED_DIR / f"{experiment}_events.pq")

        event_feature_functions = [fitts_law_get_fixation_overshoot, fitts_law_get_fixations_pr_second, get_pre_calculated_metrics_feature]
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
    
    logging.info("Finished extracting fitts law features")

    return df_features


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

def get_king_devick_features(event_features:bool, sample_features:bool) -> pd.DataFrame:
    """Runs all king devick features extractions

    Args:
        df_event (pd.DataFrame): The preprocessed event dataframe
        df_samples (pd.DataFrame): The preprocessed sample dataframe
    
    Returns:
        pd.DataFrame: Dataframe with columns ["experiment", "participant_id", X_FEATURES], where X_FEATURES is a collection of features
    """
    logging.info("Starting fitts law feature extraction")
    
    experiment = "KING_DEVICK"
    df_event_features_list=[]
    df_sample_features_list=[]
    
    if event_features:
        logging.info("Starting event feature extraction")
        df_event = pd.read_parquet(PREPROCESSED_DIR / f"{experiment}_events.pq")

        event_feature_functions = [king_devick_get_avg_mistakes_pr_trial, king_devick_get_avg_time_elapsed_pr_trial, get_pre_calculated_metrics_feature]
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
    logging.info("Finished extracting king devick features")

    return df_features    

##################
## EVIL BASTARD ##
##################

def get_distance_to_stimulus_features(df: pd.DataFrame) -> pd.DataFrame:
    features = (df
        .assign(
            distance_to_fixpoint_left = lambda x: (x["x_left"]-x["stimulus_x"])**2+(x["y_left"]-x["stimulus_y"])**2,
            distance_to_fixpoint_right = lambda x: (x["x_right"]-x["stimulus_x"])**2+(x["y_right"]-x["stimulus_y"])**2
        )
        .assign(
            distance_to_fixpoint = lambda x: 
                np.where(
                    ~x["distance_to_fixpoint_left"].isna() & ~x["distance_to_fixpoint_right"].isna(),
                    (x["distance_to_fixpoint_left"]+x["distance_to_fixpoint_right"])/2,
                
                    np.where(
                        ~x["distance_to_fixpoint_left"].isna(),
                        x["distance_to_fixpoint_left"],
                        x["distance_to_fixpoint_right"]
                    )
                )
        )
        .groupby(["experiment", "participant_id"])
        .agg({
            'distance_to_fixpoint': ["mean", "min", "max", "median", "std"],
        })
        .reset_index()
        .pipe(rename_columns)
    )
    
    return features

def get_evil_bastard_features() -> pd.DataFrame:
    """Runs all evil bastard features extractions

    Returns:
        pd.DataFrame: Dataframe with columns ["experiment", "participant_id", X_FEATURES], where X_FEATURES is a collection of features
    """

    logging.info("Extracting evil bastard features")
    
    experiment = "EVIL_BASTARD"
    
    # Read participant and trial id to identify unique participants
    df_index = pd.read_parquet(
        f"{PREPROCESSED_DIR}/{experiment}_events.pq", 
        columns=["participant_id"]
    )
    participant_groups = df_index["participant_id"].unique()
    
    df_features_all_participants = []
    for participant_id in tqdm(participant_groups, total=len(participant_groups)):
        logging.info(f"Processing participant {participant_id}")

        filters = [('participant_id', '=', participant_id)]
        df_event = pd.read_parquet(PREPROCESSED_DIR / f"{experiment}_events.pq", filters=filters)
        df_sample = (pd.read_parquet(PREPROCESSED_DIR / f'{experiment}_samples.pq', filters=filters)
        .sort_values(["experiment", "participant_id", "trial_id","time"])
        )
        df_combined = combine_samples_events(df_sample, df_event)
        
        logging.info("Starting event feature extraction")
        event_feature_functions = [get_pre_calculated_metrics_feature, get_distance_between_fixations]
        df_event_features_list = [f(df=df_event) for f in event_feature_functions]

        logging.info("Starting sample feature extraction")
        sample_feature_functions = [get_acceleration_feature, get_disconjugacy_feature]
        df_sample_features_list = [f(df=df_sample) for f in sample_feature_functions]
        
        logging.info("Starting combined feature extraction")
        combined_feature_functions = [get_distance_to_stimulus_features]
        df_combined_features_list = [f(df=df_combined) for f in combined_feature_functions]
    
        df_features_par_list = df_event_features_list + df_sample_features_list + df_combined_features_list
    
        df_features_par = reduce(lambda x, y: pd.merge(x, y, on = ["experiment", "participant_id"]), df_features_par_list)

        df_features_all_participants.append(df_features_par)
    
    df_features = pd.concat(df_features_all_participants, ignore_index=True)
    
    logging.info("Finished extracting evil bastard features")
    
    return df_features


############
## SHAPES ##
############

####################
## SMOOTH PURSUIT ##
####################

###############333
EXPERIMENT_EVENT_FEATURE_MAP = {
    "ANTI_SACCADE" : [get_pre_calculated_metrics_feature, anti_saccade_get_n_correct_trials_feature, anti_saccade_get_prop_trials_feature, anti_saccade_get_reaction_time_feature],
    "REACTION" : [get_pre_calculated_metrics_feature, reaction_get_n_correct_trials_feature, reaction_get_prop_trials_feature, reaction_get_reaction_time_feature],
    "FITTS_LAW" : [fitts_law_get_fixation_overshoot, fitts_law_get_fixations_pr_second, get_pre_calculated_metrics_feature],
    "KING_DEVICK" : [king_devick_get_avg_mistakes_pr_trial, king_devick_get_avg_time_elapsed_pr_trial, get_pre_calculated_metrics_feature],
    "EVIL_BASTARD" : [get_pre_calculated_metrics_feature, get_distance_between_fixations],
    "SHAPES" : [get_pre_calculated_metrics_feature, get_distance_between_fixations],
    "SMOOTH_PURSUITS" : [get_pre_calculated_metrics_feature, get_distance_between_fixations]
}
EXPERIMENT_SAMPLE_FEATURE_MAP = {
    "ANTI_SACCADE" : [get_acceleration_feature, get_disconjugacy_feature],
    "REACTION" : [get_acceleration_feature, get_disconjugacy_feature],
    "FITTS_LAW" : [get_acceleration_feature, get_disconjugacy_feature],
    "KING_DEVICK" : [get_acceleration_feature, get_disconjugacy_feature],
    "EVIL_BASTARD" : [get_acceleration_feature, get_disconjugacy_feature],
    "SHAPES" : [get_acceleration_feature, get_disconjugacy_feature],
    "SMOOTH_PURSUITS" : [get_acceleration_feature, get_disconjugacy_feature]
}
EXPERIMENT_COMBINED_FEATURE_MAP = {
    "EVIL_BASTARD" : [get_distance_to_stimulus_features],
    "SHAPES" : [get_distance_to_stimulus_features],
    "SMOOTH_PURSUITS" : [get_distance_to_stimulus_features]
}

def get_features(experiment) -> pd.DataFrame:
    """Runs all features extractions

    Returns:
        pd.DataFrame: Dataframe with columns ["experiment", "participant_id", X_FEATURES], where X_FEATURES is a collection of features
    """

    logging.info("Extracting features")
    
    # Read participant and trial id to identify unique participants
    df_index = pd.read_parquet(
        f"{PREPROCESSED_DIR}/{experiment}_events.pq", 
        columns=["participant_id"]
    )
    participant_groups = df_index["participant_id"].unique()
    
    df_features_all_participants = []
    for participant_id in tqdm(participant_groups, total=len(participant_groups)):
        logging.info(f"Processing participant {participant_id}")

        filters = [('participant_id', '=', participant_id)]
        df_event = pd.read_parquet(PREPROCESSED_DIR / f"{experiment}_events.pq", filters=filters)
        df_sample = (pd.read_parquet(PREPROCESSED_DIR / f'{experiment}_samples.pq', filters=filters)
        .sort_values(["experiment", "participant_id", "trial_id","time"])
        )
        df_combined = combine_samples_events(df_sample, df_event)
        
        logging.info("Starting event feature extraction")
        event_feature_functions = EXPERIMENT_EVENT_FEATURE_MAP[experiment]
        df_event_features_list = [f(df=df_event) for f in event_feature_functions]

        logging.info("Starting sample feature extraction")
        sample_feature_functions = EXPERIMENT_SAMPLE_FEATURE_MAP[experiment]
        df_sample_features_list = [f(df=df_sample) for f in sample_feature_functions]
        
        if experiment in EXPERIMENT_COMBINED_FEATURE_MAP:
            logging.info("Starting combined feature extraction")
            combined_feature_functions = EXPERIMENT_COMBINED_FEATURE_MAP[experiment]
            df_combined_features_list = [f(df=df_combined) for f in combined_feature_functions]
        
            df_features_par_list = df_event_features_list + df_sample_features_list + df_combined_features_list
        else:
            df_features_par_list = df_event_features_list + df_sample_features_list
    
        df_features_par = reduce(lambda x, y: pd.merge(x, y, on = ["experiment", "participant_id"]), df_features_par_list)

        df_features_all_participants.append(df_features_par)
    
    df_features = pd.concat(df_features_all_participants, ignore_index=True)
    
    logging.info("Finished extracting evil bastard features")
    
    return df_features





