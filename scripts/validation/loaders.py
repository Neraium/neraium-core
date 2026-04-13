"""Dataset loaders for different formats."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


class DatasetLoader:
    """Base class for dataset loaders."""

    @staticmethod
    def load(path: Path) -> pd.DataFrame:
        """Load dataset from path. Must return DataFrame with standardized columns."""
        raise NotImplementedError


class FDDatasetLoader(DatasetLoader):
    """Loader for FD001/FD004 CMAPSS datasets."""

    @staticmethod
    def load(path: Path) -> pd.DataFrame:
        """Load FD CMAPSS dataset."""
        df = pd.read_csv(path)

        # Ensure required columns exist
        required = ["unit", "cycle"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"FD dataset missing columns: {missing}")

        # Normalize drift score column names
        drift_cols = [c for c in df.columns if "drift" in c.lower()]
        if not drift_cols:
            # If no drift score, create default
            df["structural_drift_score"] = 0.0

        # Create standardized columns if missing
        if "failure_cycle" not in df.columns and "true_rul" in df.columns:
            # Estimate failure cycle from RUL
            df["failure_cycle"] = df.groupby("unit")["cycle"].transform("max") + df["true_rul"]

        if "lead_time" not in df.columns:
            if "lead_time_hours" in df.columns:
                df["lead_time"] = df["lead_time_hours"]
            elif "alert_lead" in df.columns:
                df["lead_time"] = df["alert_lead"]
            else:
                df["lead_time"] = None

        return df


class IMSDatasetLoader(DatasetLoader):
    """Loader for IMS bearing datasets."""

    @staticmethod
    def load(path: Path) -> pd.DataFrame:
        """Load IMS dataset."""
        df = pd.read_csv(path)

        # IMS datasets have different column names
        # Standardize them
        if "drift_smooth" in df.columns:
            df["structural_drift_score"] = df["drift_smooth"]
        elif "drift" in df.columns:
            df["structural_drift_score"] = df["drift"]
        else:
            df["structural_drift_score"] = 0.0

        # Create time-based cycle if not present
        if "cycle" not in df.columns:
            if "t" in df.columns:
                df["cycle"] = df["t"]
            else:
                df["cycle"] = range(len(df))

        # Create unit identifier if not present
        if "unit" not in df.columns:
            if "file_name" in df.columns:
                # Use file_name as unit identifier
                df["unit"] = df["file_name"]
            else:
                df["unit"] = 1

        return df


class GenericDatasetLoader(DatasetLoader):
    """Loader for generic CSV datasets."""

    @staticmethod
    def load(path: Path) -> pd.DataFrame:
        """Load generic CSV dataset."""
        df = pd.read_csv(path)

        # Ensure basic columns
        if "unit" not in df.columns:
            if "id" in df.columns:
                df["unit"] = df["id"]
            else:
                df["unit"] = 1

        if "cycle" not in df.columns:
            if "time" in df.columns:
                df["cycle"] = df["time"]
            elif "timestamp" in df.columns:
                df["cycle"] = range(len(df))
            else:
                df["cycle"] = range(len(df))

        # Normalize drift score
        if "structural_drift_score" not in df.columns:
            drift_cols = [c for c in df.columns if "drift" in c.lower() and "score" in c.lower()]
            if drift_cols:
                df["structural_drift_score"] = df[drift_cols[0]]
            else:
                df["structural_drift_score"] = 0.0

        return df


def load_dataset(
    path: Path,
    dataset_type: str,
) -> Optional[pd.DataFrame]:
    """
    Load a dataset from path.

    Args:
        path: Path to dataset CSV file
        dataset_type: One of 'fd001', 'fd004', 'ims', 'igrow'

    Returns:
        DataFrame or None if file not found
    """
    if not path.exists():
        return None

    try:
        if dataset_type in ("fd001", "fd004"):
            return FDDatasetLoader.load(path)
        elif dataset_type == "ims":
            return IMSDatasetLoader.load(path)
        else:
            # Generic loader for igrow or unknown types
            return GenericDatasetLoader.load(path)
    except Exception as e:
        print(f"Error loading {dataset_type} from {path}: {e}")
        return None
