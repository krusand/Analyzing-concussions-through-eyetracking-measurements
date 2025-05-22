# Investigating Concussion Subtypes with Eye-Tracking Data
This repository contains the code for our Bachelor thesis at the IT University of Copenhagen. 

## Data

Our data was provided by the VISCOM project. We cannot provide the data, but others could redo the experiments, to get the same data structure. 

## Setting up the environment

To set up the virtual environment, go to the project directory and run

```zsh
python3 -m venv .venv
source .venv/bin/activate
```

After, run 

```zsh
pip install -r requirements.txt
```

## Running scripts

The pipeline from data extraction until feature extraction is ran using the script `src/pipeline.py`. To run the script for all tasks, run:

```zsh
python3 src/pipeline.py --all_experiments
```
To run for specific experiments, use:

```
python3 src/pipeline.py --experiments EXPERIMENT1 EXPERIMENT2
```

## Running clustering

The two feature selection and clustering methods described can be found in the scripts `src/supervised_method_clustering.ipynb` and `src/unsupervised_method_clustering.ipynb`. 










