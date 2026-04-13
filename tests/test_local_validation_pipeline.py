"""Tests for the local validation pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.validation.config import ValidationConfig, discover_local_paths
from scripts.validation.loaders import load_dataset, FDDatasetLoader, IMSDatasetLoader, GenericDatasetLoader
from scripts.validation.metrics import compute_metrics, create_lead_time_summary, compute_lead_time
from scripts.validation.plots import create_unit_plot, generate_plots


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def fd_sample_data():
    """Create sample FD004-like dataset."""
    data = {
        "unit": [1, 1, 1, 2, 2, 2, 3, 3, 3] * 20,
        "cycle": list(range(1, 21)) * 9,
        "structural_drift_score": np.random.random(180) * 0.5,
        "failure_cycle": [100, 100, 100, 150, 150, 150, 80, 80, 80] * 20,
    }
    df = pd.DataFrame(data)
    # Make scores increase over time per unit
    for unit in df["unit"].unique():
        mask = df["unit"] == unit
        indices = np.where(mask)[0]
        df.loc[indices, "structural_drift_score"] = np.linspace(0, 0.7, len(indices))
    return df


@pytest.fixture
def ims_sample_data():
    """Create sample IMS-like dataset."""
    return pd.DataFrame({
        "t": range(1, 101),
        "file_name": ["2004.02.12.10.32.39"] * 100,
        "drift": np.linspace(0, 0.6, 100),
        "drift_smooth": np.linspace(0, 0.6, 100),
        "state": ["STABLE"] * 50 + ["DEGRADED"] * 50,
    })


@pytest.fixture
def generic_sample_data():
    """Create generic sample dataset."""
    return pd.DataFrame({
        "unit": [1] * 50 + [2] * 50,
        "cycle": list(range(50)) * 2,
        "drift": np.concatenate([
            np.linspace(0, 0.4, 50),
            np.linspace(0, 0.6, 50),
        ]),
    })


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Config Tests
# ============================================================================


def test_validation_config_defaults():
    """Test ValidationConfig default values."""
    config = ValidationConfig()
    assert config.output_dir == Path("validation_output")
    assert config.drift_threshold == 0.5
    assert config.dry_run is False


def test_validation_config_get_path():
    """Test ValidationConfig.get_path()."""
    test_path = Path("/tmp/test.csv")
    config = ValidationConfig(fd004_path=test_path)
    assert config.get_path("fd004") == test_path
    assert config.get_path("fd001") is None


def test_discover_local_paths():
    """Test path discovery returns dict with all datasets."""
    paths = discover_local_paths()
    assert isinstance(paths, dict)
    assert set(paths.keys()) == {"fd001", "fd004", "ims", "igrow"}


# ============================================================================
# Loader Tests
# ============================================================================


def test_fd_loader(fd_sample_data, temp_output_dir):
    """Test FD dataset loader."""
    csv_path = temp_output_dir / "test_fd.csv"
    fd_sample_data.to_csv(csv_path, index=False)

    df = FDDatasetLoader.load(csv_path)
    assert df is not None
    assert "unit" in df.columns
    assert "cycle" in df.columns
    assert "structural_drift_score" in df.columns


def test_ims_loader(ims_sample_data, temp_output_dir):
    """Test IMS dataset loader."""
    csv_path = temp_output_dir / "test_ims.csv"
    ims_sample_data.to_csv(csv_path, index=False)

    df = IMSDatasetLoader.load(csv_path)
    assert df is not None
    assert "unit" in df.columns
    assert "cycle" in df.columns
    assert "structural_drift_score" in df.columns


def test_generic_loader(generic_sample_data, temp_output_dir):
    """Test generic dataset loader."""
    csv_path = temp_output_dir / "test_generic.csv"
    generic_sample_data.to_csv(csv_path, index=False)

    df = GenericDatasetLoader.load(csv_path)
    assert df is not None
    assert "unit" in df.columns
    assert "cycle" in df.columns


def test_load_dataset_missing_file():
    """Test loading non-existent file."""
    result = load_dataset(Path("/nonexistent/file.csv"), "fd004")
    assert result is None


def test_load_dataset_invalid_type(generic_sample_data, temp_output_dir):
    """Test loading with generic loader."""
    csv_path = temp_output_dir / "test.csv"
    generic_sample_data.to_csv(csv_path, index=False)
    result = load_dataset(csv_path, "igrow")
    assert result is not None


# ============================================================================
# Metrics Tests
# ============================================================================


def test_compute_lead_time_below_threshold():
    """Test lead time when drift never exceeds threshold."""
    df = pd.DataFrame({
        "cycle": range(100),
        "structural_drift_score": np.linspace(0, 0.3, 100),
    })
    lead_time = compute_lead_time(df, drift_threshold=0.5)
    assert lead_time == 100  # Full cycle count


def test_compute_lead_time_above_threshold():
    """Test lead time when drift exceeds threshold."""
    df = pd.DataFrame({
        "cycle": range(100),
        "structural_drift_score": np.linspace(0, 1.0, 100),
    })
    lead_time = compute_lead_time(df, drift_threshold=0.5)
    # Should exceed at ~50% of 100
    assert 40 < lead_time < 60


def test_compute_metrics_basic(fd_sample_data):
    """Test basic metric computation."""
    metrics = compute_metrics(fd_sample_data, "FD004")
    assert metrics.dataset_name == "FD004"
    assert metrics.unit_count == 3
    assert metrics.total_records == 180
    assert metrics.mean_lead_time > 0
    assert metrics.median_lead_time > 0
    assert 0 <= metrics.p50_gt_50_cycles <= 100
    assert 0 <= metrics.p50_gt_100_cycles <= 100


def test_compute_metrics_invalid_input():
    """Test metrics computation with empty data."""
    with pytest.raises(ValueError):
        compute_metrics(pd.DataFrame(), "TEST")


def test_metrics_to_dict(fd_sample_data):
    """Test metrics serialization to dict."""
    metrics = compute_metrics(fd_sample_data, "FD004")
    d = metrics.to_dict()
    assert "dataset" in d
    assert "unit_count" in d
    assert "mean_lead_time" in d
    assert isinstance(d["mean_lead_time"], float)


def test_create_lead_time_summary(fd_sample_data):
    """Test lead time summary creation."""
    metrics = compute_metrics(fd_sample_data, "FD004")
    summary = create_lead_time_summary(fd_sample_data, metrics)
    assert len(summary) == 3  # 3 units
    assert "unit" in summary.columns
    assert "lead_time" in summary.columns
    assert "is_best_case" in summary.columns


# ============================================================================
# Plot Tests
# ============================================================================


def test_create_unit_plot_basic(fd_sample_data, temp_output_dir):
    """Test basic plot creation."""
    unit_data = fd_sample_data[fd_sample_data["unit"] == 1]
    fig, ax = create_unit_plot(unit_data, 1, "FD004", "best")
    assert fig is not None
    assert ax is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_generate_plots_output(fd_sample_data, temp_output_dir):
    """Test plot generation creates files."""
    metrics = compute_metrics(fd_sample_data, "FD004")
    output_dir = temp_output_dir / "plots"

    paths = generate_plots(
        fd_sample_data,
        "FD004",
        metrics.best_case_unit,
        metrics.median_case_unit,
        metrics.worst_case_unit,
        output_dir,
    )

    assert len(paths) > 0
    for path in paths:
        assert path.exists()
        assert path.suffix == ".png"


def test_generate_plots_missing_unit():
    """Test plot generation with missing unit."""
    df = pd.DataFrame({
        "unit": [1, 1],
        "cycle": [1, 2],
        "structural_drift_score": [0.1, 0.2],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "plots"
        # Should handle missing units gracefully
        paths = generate_plots(df, "TEST", 999, 999, 999, output_dir)
        assert len(paths) == 0


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_pipeline_fd(fd_sample_data, temp_output_dir):
    """Test full pipeline with FD data."""
    # Save sample data
    csv_path = temp_output_dir / "fd_data.csv"
    fd_sample_data.to_csv(csv_path, index=False)

    # Load
    df = load_dataset(csv_path, "fd004")
    assert df is not None

    # Compute metrics
    metrics = compute_metrics(df, "FD004")
    assert metrics.unit_count == 3

    # Create summary
    summary = create_lead_time_summary(df, metrics)
    assert len(summary) == 3

    # Generate plots
    plots_dir = temp_output_dir / "plots"
    paths = generate_plots(
        df, "FD004",
        metrics.best_case_unit,
        metrics.median_case_unit,
        metrics.worst_case_unit,
        plots_dir,
    )
    assert len(paths) > 0


def test_missing_dataset_gracefully():
    """Test that missing datasets are handled gracefully."""
    config = ValidationConfig(
        fd001_path=Path("/nonexistent/fd001.csv"),
    )
    path = config.get_path("fd001")
    assert path is not None
    assert not path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
