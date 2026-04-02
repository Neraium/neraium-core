"""
Configuration for Neraium Markets
"""

from pathlib import Path

# Core paths
DATA_DIR = Path("sample_data")

# Column names
DATE_COLUMN = "timestamp"
PRICE_COLUMN = "close"

REQUIRED_COLUMNS = frozenset(
    {DATE_COLUMN, "open", "high", "low", PRICE_COLUMN, "volume"}
)

# Assets
ASSETS = [
    "spy", "qqq", "iwm",
    "xlf", "xlk", "xle", "xlv", "xli", "xly", "xlp", "xlu",
    "vix", "dxy", "gold", "oil", "us2y", "us10y"
]

# Groups
CORE_EQUITY_ASSETS = ["spy", "qqq", "iwm"]

SECTOR_ASSETS = [
    "xlf", "xlk", "xle", "xlv",
    "xli", "xly", "xlp", "xlu"
]

CROSS_ASSET_CONTEXT = [
    "vix", "dxy", "gold", "oil", "us2y", "us10y"
]
