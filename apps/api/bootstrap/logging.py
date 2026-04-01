from __future__ import annotations

import logging
import os


def configure_logging() -> None:
    raw = str(os.getenv("NERAIUM_LOG_LEVEL", "INFO")).strip().upper()
    level = getattr(logging, raw, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
