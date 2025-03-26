import pandas as pd
import os
from tqdm import tqdm
from config import *

def convert_asc_to_pq():
    colnames = [
        "time",
        "x_left",
        "y_left",
        "pupil_size_left",
        "x_right",
        "y_right",
        "pupil_size_right",
        "x_velocity_left",
        "y_velocity_left",
        "x_velocity_right",
        "y_velocity_rigth",
        "x_resolution",
        "y_resolution",
        ".",
        "error_message"]
    
    file_filters = ["anti-saccade", "FittsLaw", "Fixations", "KingDevick", "Patterns", "Reaction", "Shapes", "SmoothPursuits"]
    experiments = ["ANTI_SACCADE" , "FITTS_LAW", "FIXATIONS", "KING_DEVICK", "EVIL_BASTARD", "REACTION", "SHAPES", "SMOOTH_PURSUITS"]
    for file_filter, experiment in zip(file_filters, experiments):
        asc_files = [f for f in os.listdir(ASC_RAW_SAMPLES_DIR) if f.endswith('.asc') and f.startswith(f"{file_filter}")]

        samples_dfs = []
        for file_path in tqdm(asc_files):
            df = pd.read_csv(file_path, names=colnames, delimiter="\t", skipinitialspace=True)        
            for col in df.columns:
                if col not in [".", "error_message"]:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            samples_dfs.append(df)
        
        path_save = PROCESSED_DIR / f"{experiment}_SAMPLES.pq"
        print(f"Saving to {path_save}")
        samples_dfs.to_parquet(path_save, index=False)

def main():
    # Convert asc files to parquet files
    convert_asc_to_pq()

if __name__ == '__main__':
    main()
