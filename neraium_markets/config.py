from pathlib import Path

ASSETS = [
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
]

DATA_DIR = Path(__file__).resolve().parent / "sample_data"
DATE_COLUMN = "timestamp"
PRICE_COLUMN = "close"
