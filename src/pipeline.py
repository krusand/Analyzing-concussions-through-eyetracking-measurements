#!/usr/bin/env python3
"""
Eye Tracking Data Processing Pipeline

This script orchestrates the complete eye tracking data processing workflow
by running each processing step in sequence.
"""

import sys
import subprocess
from pathlib import Path
from config import *

experiments = ["EVIL_BASTARD"] #["ANTI_SACCADE", "FITTS_LAW", "FIXATIONS", "KING_DEVICK", "EVIL_BASTARD", "REACTION", "SHAPES", "SMOOTH_PURSUITS"]
file_filters = ["Patterns"] #["anti-saccade", "FittsLaw", "Fixations", "KingDevick", "Patterns", "Reaction", "Shapes", "SmoothPursuits"]

# Define the pipeline steps as ordered scripts to run
EVENTS_PIPELINE_STEPS = [
    # { # --> RAW
    #     "name": "Extracting Events",
    #     "script": "extracting_events.py",
    #     "description": "Converts asc files to parquet files",
    #     "args": {
    #             "experiments" : experiments,
    #             "file_filters" : file_filters
    #     }
    # },
    { # RAW --> CLEANED
        "name": "Cleaning Events",
        "script": "cleaning_events.py",
        "description": "Remove invalid trials and participants",
        "args": {
                "experiments" : experiments
        }
    },
    { # CLEANED --> PREPROCESSED
        "name": "Preprocessing Events",
        "script": "preprocessing_events.py",
        "description": "Applies general and experiment-specific preprocessing",
        "args": {
                "experiments" : experiments
        }
    },
    # {
    #     "name": "Feature Extraction",
    #     "script": "extract_features.py",
    #     "description": "Extracts relevant features from preprocessed data"
    # },
    # {
    #     "name": "Analysis",
    #     "script": "analyze_data.py",
    #     "description": "Performs statistical analysis on extracted features"
    # },
    # {
    #     "name": "Visualization",
    #     "script": "visualize_results.py",
    #     "description": "Generates plots and visualizations of results"
    # }
]

SAMPLES_PIPELINE_STEPS = [
    # { # --> CLEANED
    #     "name": "Extracting Samples",
    #     "script": "extracting_samples.py",
    #     "description": "Converts asc files to parquet files",
    #     "args": {
    #             "experiments" : experiments,
    #             "file_filters" : file_filters
    #     }
    # },
    { # --> CLEANED
        "name": "Preprocessing Samples",
        "script": "preprocessing_samples.py",
        "description": "Applies general transformations to sample data",
        "args": {
                "experiments" : experiments
        }
    },
    # { # CLEANED --> PREPROCESSED
    #     "name": "Preprocessing",
    #     "script": "preprocessing_samples.py",
    #     "description": "Preprocessing of sample data"
    # }
]


def run_script(script_path, args=None):    
    command = [sys.executable, str(script_path)]
    
    if args:
        if "experiments" in args:
            command += ["--experiments"] + args["experiments"]
        if "file_filters" in args:
            command += ["--file_filters"] + args["file_filters"]
    
    try:
        # Run the script using the same Python interpreter as the current process
        result = subprocess.run(
            command,
            check=True,
            text=True
        )
        print(f"Finished: {script_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        return False

def run_pipeline():
    """Run the complete data processing pipeline"""
    
    print("Starting pipeline")
    
    # Get the directory where the pipeline script is located
    pipeline_dir = Path(__file__).parent.absolute()
      
    # Run each step in sequence
    for step in EVENTS_PIPELINE_STEPS:
        script_path = pipeline_dir / step["script"]
        print(f"\nStep: {step['name']}")
        print(f"Description: {step['description']}")
        if not script_path.exists():
            print(f"Script not found: {script_path}")
            continue
        run_script(script_path, step.get("args"))
    
    # Run each step in sequence
    for step in SAMPLES_PIPELINE_STEPS:
        script_path = pipeline_dir / step["script"]
        print(f"\nStep: {step['name']}")
        print(f"Description: {step['description']}")
        if not script_path.exists():
            print(f"Script not found: {script_path}")
            continue
        run_script(script_path, step.get("args"))

if __name__ == "__main__":
    run_pipeline()