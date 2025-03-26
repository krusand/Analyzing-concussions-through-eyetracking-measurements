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
        "y_velocity_right",
        "x_resolution",
        "y_resolution",
        ".",
        "error_message"]
    
    file_filters = ["anti-saccade", "FittsLaw", "Fixations", "KingDevick", "Patterns", "Reaction", "Shapes", "SmoothPursuits"]
    experiments = ["ANTI_SACCADE" , "FITTS_LAW", "FIXATIONS", "KING_DEVICK", "EVIL_BASTARD", "REACTION", "SHAPES", "SMOOTH_PURSUITS"]
    for file_filter, experiment in zip(file_filters, experiments):
        asc_files = [f for f in os.listdir(ASC_RAW_SAMPLES_DIR) if f.endswith('.asc') and f.startswith(f"{file_filter}")]

        samples = []
        for file_name in tqdm(asc_files):
            file_path = ASC_RAW_SAMPLES_DIR / file_name
            df = pd.read_csv(
                            file_path, 
                            names=colnames, 
                            delimiter="\t", 
                            skipinitialspace=True,
                            na_values={col: "." for col in [  # Apply na_values ONLY to numeric columns
                                "time", "x_left", "y_left", "pupil_size_left", "x_right", "y_right", 
                                "pupil_size_right", "x_velocity_left", "y_velocity_left", 
                                "x_velocity_right", "y_velocity_right", "x_resolution", "y_resolution"
                            ]},
                            dtype = {"time": int,
                                      "x_left": float,
                                      "y_left": float,
                                      "pupil_size_left": float,
                                      "x_right": float,
                                      "y_right": float,
                                      "pupil_size_right": float,
                                      "x_velocity_left": float,
                                      "y_velocity_left": float,
                                      "x_velocity_right": float,
                                      "y_velocity_right": float,
                                      "x_resolution": float,
                                      "y_resolution": float,
                                      ".": str,
                                      "error_message": str})
            df = df.drop(".", axis=1)        
            for col in df.columns:
                if col not in [".", "error_message"]:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            participant_id = file_name.split("_")[1]
            df.insert(loc=0, column='participant_id', value=participant_id)
            df.insert(loc=0, column='experiment', value=experiment)
            samples.append(df)
        
        path_save = PROCESSED_DIR / f"{experiment}_SAMPLES.pq"
        print(f"Saving to {path_save}")
        samples_df = pd.concat(samples)
        samples_df.to_parquet(path_save, index=False)

def main():
    # Convert asc files to parquet files
    convert_asc_to_pq()

if __name__ == '__main__':
    main()
