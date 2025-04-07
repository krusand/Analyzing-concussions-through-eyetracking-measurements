import pandas as pd
from config import *
import argparse
import gc
from tqdm import tqdm

###################
###   General   ###
###################

def standardise_time(df):
    
    min_time = df["time"].min()
    df.loc[:,"time"] = df["time"] - min_time
    return df

def preprocess_general(df):
    df_transformed = (
        df.pipe(standardise_time)
    )
    
    return df_transformed

def preprocess_sample(experiment):
    """Process file by participant/trial groups to reduce memory usage"""
    print(f"Starting preprocessing for {experiment}:")
    
    first_write = True
    
    # Read participant and trial id to identify unique groups
    df_index = pd.read_parquet(
        f"{CLEANED_DIR}/{experiment}_samples.pq", 
        columns=["participant_id","trial_id"]
    )

    participant_groups = df_index.groupby("participant_id")["trial_id"].unique()
        
    # Process each group
    for i, (participant_id, trial_ids) in tqdm(enumerate(participant_groups.items()), total=len(participant_groups)):
        print(f"Processing participant {participant_id}")
        
        # Read data for participant
        filters = [
            ('participant_id', '=', participant_id),
        ]
        df_participant = pd.read_parquet(
            f"{CLEANED_DIR}/{experiment}_samples.pq",
            filters=filters
            )
        
        df_chunks = []
        for trial_id in trial_ids:
            df_group = df_participant[df_participant["trial_id"]==trial_id] 
            # Preprocess each trial
            df_chunks.append(preprocess_general(df_group))
        
        df_transformed = pd.concat(df_chunks, ignore_index=True)
        
        # Write to parquet
        df_transformed.to_parquet(
            PREPROCESSED_DIR / f"{experiment}_samples.pq", 
            engine="fastparquet", 
            append = not first_write)
        first_write = False
        
        # Clean up to free memory
        del df_participant, df_chunks, df_transformed
        gc.collect()
    
def main(experiments):
    # Convert asc files to parquet files
    for experiment in experiments:
        preprocess_sample(experiment)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract events from ASC files.")
    parser.add_argument("--experiments", nargs='+', required=True, help="List of experiment names")
    args = parser.parse_args()
    
    main(args.experiments)