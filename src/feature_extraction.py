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
    
    df_features = reduce(lambda x, y: pd.merge(x, y, on = ["participant_id"], how="outer"), df_features_list)
    
    logging.info("Finished loading features")
    
    return df_features

def load_demographic_info() -> pd.DataFrame:
    logging.info("Loading demographics")
    demographics = pd.read_excel(DATA_DIR / "demographic_info.xlsx")
    
    # Filter
    demographics = demographics[demographics["Eye tracking date"].notna()]
    
    # Mutate
    demographics["y"] = (demographics["Group"] == "PATIENT").astype(int)
    demographics["participant_id"] = demographics["ID"].astype(int)
    
    # Select
    demographics = demographics[["participant_id", "y"]]
    return demographics


def join_features_on_demographic_info(feature_df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Joining features on demographics")
    demographics = load_demographic_info()
    return pd.merge(demographics, feature_df, how='left', on='participant_id')
    
    
def run_feature_extraction(args: argparse.ArgumentParser) -> None:
    logging.info("Running feature extraction")
    experiments = args.experiments
    event_features = args.event_features
    sample_features = args.sample_features
    
    for experiment in experiments:
        features = get_features(experiment, event_features, sample_features)
        features.to_parquet(FEATURES_DIR / f"{experiment}_features.pq")
        logging.info(f"Saved {experiment} features to file")


def main(args: argparse.ArgumentParser) -> None:
    run_feature_extraction(args)
    # features = load_features(args.experiments)
    # data = join_features_on_demographic_info(feature_df=features)
    # data.to_parquet(FEATURES_DIR / 'features.pq')
    # logging.info("Saved features to file")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run feature extraction")
    parser.add_argument("--experiments", nargs='+', required=True, help="List of experiment names")
    parser.add_argument("--event_features", action=argparse.BooleanOptionalAction, required=False, help="Should event features be calculated")
    parser.add_argument("--sample_features", action=argparse.BooleanOptionalAction, required=False, help="Should sample features be calculated")
    args = parser.parse_args()    

    if args.event_features is None and args.sample_features is None:
        logging.info("Defaulting to extracting both sample and event features")
        args.event_features = True
        args.sample_features = True
    
    main(args)
