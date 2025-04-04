#!/usr/bin/env python3
"""
Eye Tracking Data Processing Pipeline

This script orchestrates the complete eye tracking data processing workflow
by running each processing step in sequence.
"""

import os
import sys
import subprocess
import logging
import time
from datetime import datetime
from pathlib import Path
from config import *

# Configure logging
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("eye_tracking_pipeline")

# Define the pipeline steps as ordered scripts to run
EVENTS_PIPELINE_STEPS = [
    { # --> RAW
        "name": "Extracting Events",
        "script": "extracting_events.py",
        "description": "Converts asc files to parquet files"
    },
    { # RAW --> CLEANED
        "name": "Cleaning Events",
        "script": "cleaning_events.py",
        "description": "Remove invalid trials and participants"
    },
    { # CLEANED --> PREPROCESSED
        "name": "Preprocessing",
        "script": "preprocessing.py",
        "description": "Applies general and experiment-specific preprocessing"
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
    { # --> CLEANED
        "name": "Extracting Samples",
        "script": "extracting_events.py",
        "description": "Converts asc files to parquet files"
    },
    { # CLEANED --> PREPROCESSED
        "name": "Preprocessing",
        "script": "preprocessing_samples.py",
        "description": "Preprocessing of sample data"
    }
]


def run_script(script_path):
    """
    Run a Python script and handle errors
    
    Args:
        script_path: Path to the Python script to run
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Running {script_path}")
    start_time = time.time()
    
    try:
        # Run the script using the same Python interpreter as the current process
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            text=True,
            capture_output=True
        )
        
        # Log the standard output
        if result.stdout:
            logger.info(f"Output from {script_path}:\n{result.stdout}")
            
        elapsed_time = time.time() - start_time
        logger.info(f"Successfully completed {script_path} in {elapsed_time:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_path}: {e}")
        if e.stdout:
            logger.info(f"Standard output:\n{e.stdout}")
        if e.stderr:
            logger.error(f"Standard error:\n{e.stderr}")
        return False
        
    except Exception as e:
        logger.error(f"Unexpected error running {script_path}: {e}")
        return False

def run_pipeline():
    """Run the complete data processing pipeline"""
    logger.info("Starting eye tracking data processing pipeline")
    
    # Get the directory where the pipeline script is located
    pipeline_dir = Path(__file__).parent.absolute()
    
    # Track successful and failed steps
    successful_steps = []
    failed_steps = []
    
    # Run each step in sequence
    for step in PIPELINE_STEPS:
        script_path = pipeline_dir / step["script"]
        
        logger.info(f"=== Starting step: {step['name']} ===")
        logger.info(f"Description: {step['description']}")
        
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            failed_steps.append(step["name"])
            continue
            
        success = run_script(script_path)
        
        if success:
            successful_steps.append(step["name"])
        else:
            failed_steps.append(step["name"])
            logger.warning(f"Step '{step['name']}' failed. Continuing with next step.")
    
    # Log summary
    logger.info("\n=== Pipeline Summary ===")
    logger.info(f"Total steps: {len(PIPELINE_STEPS)}")
    logger.info(f"Successful steps: {len(successful_steps)}")
    logger.info(f"Failed steps: {len(failed_steps)}")
    
    if failed_steps:
        logger.warning(f"Failed steps: {', '.join(failed_steps)}")
        return False
    else:
        logger.info("All pipeline steps completed successfully!")
        return True

if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)