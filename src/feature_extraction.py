import os
import sys

# Local
from config import *
from features.feature_utils import *

# Other
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from functools import reduce
import argparse


def load_features(experiments: list[str]) -> pd.DataFrame:
    logging.info("Loading features")
    df_features_list = []
    
    for experiment in experiments:
        df = pd.read_parquet(FEATURES_DIR / f"{experiment}_features.pq")
        df.columns = [f'{experiment}_{column}' if column not in ['experiment', 'participant_id'] else f'{column}' for column in df.columns]
        df = df.drop("experiment", axis=1)
        df_features_list.append(df)
    
    df_features = reduce(lambda x, y: pd.merge(x, y, on = ["participant_id"]), df_features_list)
    
    logging.info("Finished loading features")
    
    return df_features

def load_demographic_info() -> pd.DataFrame:
    logging.info("Loading demographics")
    demographics = pd.read_excel(DATA_DIR / "demographic_info.xlsx")[["ID", "Group"]]

    demographics["y"] = (demographics["Group"] == "PATIENT").astype(int)
    demographics["participant_id"] = demographics["ID"].astype(int)
    demographics = demographics[["participant_id", "y"]]
    return demographics


def join_demographic_info_on_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Joining demographics on features")
    demographics = load_demographic_info()
    return pd.merge(feature_df, demographics, how='left', on='participant_id')
    
    
def run_feature_extraction(experiments: list[str]) -> None:
    logging.info("Running feature extraction")
    for experiment in experiments:
            
        df_event = pd.read_parquet(PREPROCESSED_DIR / f"{experiment}_events.pq").reset_index(drop=True)
        df_sample = (pd.read_parquet(PREPROCESSED_DIR / f'{experiment}_samples.pq')
        .sort_values(["experiment", "participant_id", "trial_id","time"])
        )
        func_name = f"get_{experiment.lower()}_features"
        feature_func = globals().get(func_name)
        if feature_func:
            features = feature_func(df_event, df_sample)
            features.to_parquet(FEATURES_DIR / f"{experiment}_features.pq")
        else:
            raise ValueError(f"No feature extraction function found for: {experiment}")


def main(experiments: list[str]) -> None:
    run_feature_extraction(experiments)
    features = load_features(experiments)
    data = join_demographic_info_on_features(feature_df=features)
    data.to_parquet(FEATURES_DIR / 'features.pq')
    logging.info("Saved features to file")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run feature extraction")
    parser.add_argument("--experiments", nargs='+', required=True, help="List of experiment names")
    args = parser.parse_args()
    main(args.experiments)
