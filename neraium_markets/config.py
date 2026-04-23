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

# Day 8: cross-asset clustering / propagation (equity ETFs only)
DAY8_CLUSTER_ASSETS = CORE_EQUITY_ASSETS + SECTOR_ASSETS

# Map config asset tickers to coarse sector buckets for influence aggregation
ASSET_TO_SECTOR: dict[str, str] = {
    "spy": "core_large",
    "qqq": "core_tech",
    "iwm": "core_small",
    "xlf": "sector_financials",
    "xlk": "sector_tech",
    "xle": "sector_energy",
    "xlv": "sector_health",
    "xli": "sector_industrial",
    "xly": "sector_consumer_disc",
    "xlp": "sector_consumer_staples",
    "xlu": "sector_utilities",
}


# Day 3 structural configuration
STRUCTURE_ASSETS = [
    "spy", "qqq", "iwm", "xlf", "xlk", "xle", "xlv", "xli", "xly", "xlp", "xlu", "vix", "dxy", "gold", "oil"
]
LEAD_LAG_PAIRS = [
    ("spy", "vix"), ("spy", "qqq"), ("spy", "dxy"), ("spy", "us10y"), ("xlk", "qqq"), ("xle", "oil")
]
BASELINE_WINDOW = 60
RECENT_WINDOW = 20
MAX_LAG = 3
