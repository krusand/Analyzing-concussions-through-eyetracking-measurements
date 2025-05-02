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
import argparse


def get_pipeline_steps(experiments: list[str], file_filters: list[str]) -> list[dict]:
    """

    Args:
        experiments (list[str]): list of experiments to run
        file_filters (list[str]): list of file_filters to use when running

    Returns:
        list[dict]: contains the scripts of the pipeline
    """

    pipeline_steps = [
        #### EVENTS:

        { # --> RAW
            "name": "Extracting Events",
            "script": "extracting_events.py",
            "description": "Converts asc files to parquet files",
            "args": {
                    "experiments" : experiments,
                    "file_filters" : file_filters
            }
        },
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
        
        #### SAMPLES: 
        
        { # --> CLEANED
            "name": "Extracting Samples",
            "script": "extracting_samples.py",
            "description": "Converts asc files to parquet files",
            "args": {
                    "experiments" : experiments,
                    "file_filters" : file_filters
            }
        },
        { # CLEANED --> PREPROCESSED
            "name": "Preprocessing Samples",
            "script": "preprocessing_samples.py",
            "description": "Applies general transformations to sample data",
            "args": {
                    "experiments" : experiments
            }
        },
        
        #### COMBINED: 
        
        { # PREPROCESSED --> FEATURE_EXTRACTION
            "name": "Feature Extraction",
            "script": "feature_extraction.py",
            "description": "Extracts relevant features from preprocessed data",
                    "args": {
                    "experiments" : experiments
            }
        },
        
        { # FEATURE_EXTRACTION --> FEATURE_SELECTION
            "name": "Feature Selection",
            "script": "feature_selection.py",
            "description": "Selects features based on different methods",
                    "args": {
                    "experiments" : experiments
            }
        }
    ]
    
    return pipeline_steps

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
        logging.info(f"Finished: {script_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running {script_path}: {e}")
        return False

def run_pipeline(experiments, file_filters):
    """Run the complete data processing pipeline"""
    
    logging.info("Starting pipeline")
    
    # Get the directory where the pipeline script is located
    pipeline_dir = Path(__file__).parent.absolute()
      
    # Run each step in sequence
    for step in get_pipeline_steps(experiments, file_filters):
        script_path = pipeline_dir / step["script"]
        logging.info(f"Step: {step['name']}")
        logging.info(f"Description: {step['description']}")
        if not script_path.exists():
            logging.warning(f"Script not found: {script_path}")
            continue
        run_script(script_path, step.get("args"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract events from ASC files.")
    parser.add_argument("--experiments", nargs='+', required=False, help="List of experiment names")
    parser.add_argument('--all_experiments', required=False, action=argparse.BooleanOptionalAction, help="Run pipeline for all experiments")
    args = parser.parse_args()
    if (not args.all_experiments or args.all_experiments is None) and (args.experiments is None):
        parser.error("--experiments must be provided when not running all experiments")

    if args.all_experiments:
        experiments = ["ANTI_SACCADE", "FITTS_LAW", "FIXATIONS", "KING_DEVICK", "EVIL_BASTARD", "REACTION", "SHAPES", "SMOOTH_PURSUITS"]
        file_filters = ["anti-saccade", "FittsLaw", "Fixations", "KingDevick", "Patterns", "Reaction", "Shapes", "SmoothPursuits"]
    else:
        experiments = args.experiments
        file_filters = [EXPERIMENT_FILE_FILTER_MAP[experiment] for experiment in experiments]
   
    
    run_pipeline(experiments, file_filters)