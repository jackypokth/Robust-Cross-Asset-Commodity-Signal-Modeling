"""
src/utils/paths.py
------------------
Central path configuration for the project.

All modules should import data paths from here rather than hardcoding
relative paths. This ensures the project works regardless of which
directory a script is run from.

Usage:
    from src.utils.paths import DATA_RAW_DIR, TRAIN_FILE, LABELS_FILE
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
LABELS_FILE: Path = DATA_RAW_DIR / "train_labels.csv"
TEST_FILE: Path = DATA_RAW_DIR / "test.csv"
TARGET_PAIRS_FILE: Path = DATA_RAW_DIR / "target_pairs.csv"

# Per-lag test label files (provided separately for evaluation)
TEST_LABELS_LAG: dict[int, Path] = {
    lag: DATA_RAW_DIR / f"test_labels_lag_{lag}.csv"
    for lag in (1, 2, 3, 4)
}

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
        {'train.csv': True, 'train_labels.csv': True, ...}
    """
    files: dict[str, Path] = {
        "train.csv": TRAIN_FILE,
        "train_labels.csv": LABELS_FILE,
        "test.csv": TEST_FILE,
        "target_pairs.csv": TARGET_PAIRS_FILE,
    }
    for lag in (1, 2, 3, 4):
        files[f"test_labels_lag_{lag}.csv"] = TEST_LABELS_LAG[lag]

    return {name: path.exists() for name, path in files.items()}


if __name__ == "__main__":
    print(f"REPO_ROOT          : {REPO_ROOT}")
    print(f"DATA_RAW_DIR       : {DATA_RAW_DIR}")
    print(f"DATA_PROCESSED_DIR : {DATA_PROCESSED_DIR}")
    print()
    print("Raw data status:")
    for name, present in check_raw_data().items():
        status = "FOUND  " if present else "MISSING"
        print(f"  {status}  {name}")
