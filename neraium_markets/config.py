"""Configuration for Neraium Markets Day 1 data pipeline."""

from pathlib import Path

from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """Validated pipeline settings (Day 1)."""

    assets: list[str] = Field(
        default_factory=lambda: [
            "spy",
            "qqq",
            "iwm",
            "xlf",
            "xlk",
            "xle",
            "xlv",
            "xli",
            "xly",
            "xlp",
            "xlu",
            "vix",
            "dxy",
            "gold",
            "oil",
            "us2y",
            "us10y",
        ],
        min_length=1,
    )
    data_dir: str = "sample_data"
    date_column: str = "timestamp"
    price_column: str = "close"

    @property
    def required_columns(self) -> tuple[str, ...]:
        return ("timestamp", "open", "high", "low", "close", "volume")

    @property
    def package_root(self) -> Path:
        return Path(__file__).resolve().parent


_SETTINGS = PipelineConfig()
ASSETS: list[str] = _SETTINGS.assets
DATA_DIR: str = _SETTINGS.data_dir
DATE_COLUMN: str = _SETTINGS.date_column
PRICE_COLUMN: str = _SETTINGS.price_column
REQUIRED_COLUMNS: tuple[str, ...] = _SETTINGS.required_columns
