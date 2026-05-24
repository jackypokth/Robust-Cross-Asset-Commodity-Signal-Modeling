"""
src/utils/paths.py
------------------
Central path configuration for the project.

All modules should import data paths from here rather than hardcoding
relative paths. This ensures the project works regardless of which
directory a script is run from.

Usage:
    from src.utils.paths import DATA_RAW_DIR, DATA_PROCESSED_DIR
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
# Resolve the project root as two levels up from this file:
#   src/utils/paths.py -> src/ -> project root
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_DIR: Path = REPO_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"

# ---------------------------------------------------------------------------
# Competition data files (raw, as downloaded from Kaggle)
# ---------------------------------------------------------------------------
TRAIN_FILE: Path = DATA_RAW_DIR / "train.csv"
TEST_FILE: Path = DATA_RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_FILE: Path = DATA_RAW_DIR / "sample_submission.csv"

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
SUBMISSIONS_DIR: Path = REPO_ROOT / "submissions"
MODELS_DIR: Path = REPO_ROOT / "models"
REPORTS_DIR: Path = REPO_ROOT / "reports"


def ensure_dirs() -> None:
    """Create all required directories if they do not exist.

    Safe to call multiple times (uses exist_ok=True).
    Useful at the start of data processing or training scripts.
    """
    for directory in [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        SUBMISSIONS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def check_raw_data() -> dict[str, bool]:
    """Check which raw data files are present.

    Returns:
        A dict mapping filename to a boolean indicating presence.

    Example:
        >>> status = check_raw_data()
        >>> print(status)
        {'train.csv': True, 'test.csv': False, 'sample_submission.csv': False}
    """
    files = {
        "train.csv": TRAIN_FILE,
        "test.csv": TEST_FILE,
        "sample_submission.csv": SAMPLE_SUBMISSION_FILE,
    }
    return {name: path.exists() for name, path in files.items()}


if __name__ == "__main__":
    print(f"REPO_ROOT          : {REPO_ROOT}")
    print(f"DATA_RAW_DIR       : {DATA_RAW_DIR}")
    print(f"DATA_PROCESSED_DIR : {DATA_PROCESSED_DIR}")
    print()
    print("Raw data status:")
    for name, present in check_raw_data().items():
        status = "FOUND" if present else "MISSING"
        print(f"  {status}  {name}")
