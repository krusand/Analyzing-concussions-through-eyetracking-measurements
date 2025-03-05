from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
ASC_DIR = DATA_DIR / "asc_files"
ASC_RAW_DIR = ASC_DIR / "raw"
METADATA_DIR = DATA_DIR / "metadata"
METADATA_RAW_DIR = METADATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RED = "255 0 0"
GREEN = "0 255 0"
BLUE = "0 0 255"
WHITE = "255 255 255"
