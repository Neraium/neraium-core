from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from .config import SIIConfig
from .engine import SIIEngine
from .errors import SIIValidationError
from .ingestion import load_frames_from_csv, load_frames_from_json
from .logging import configure_structured_logging
from .reporting import write_csv_report, write_json_report
from .types import SIIResult


@dataclass
class SIIApplication:
    """
    Deployment-oriented application wrapper around the read-only SII engine.
    """

    config: SIIConfig
    engine: SIIEngine
    logger: logging.Logger

    @classmethod
    def from_config(cls, config: SIIConfig) -> "SIIApplication":
        logger = configure_structured_logging(config.log_level)
        engine = SIIEngine(config=config)
        return cls(config=config, engine=engine, logger=logger)

    @classmethod
    def from_env(cls) -> "SIIApplication":
        return cls.from_config(SIIConfig.from_env())

    def run_payloads(self, payloads: list[dict[str, Any]]) -> list[SIIResult]:
        outputs: list[SIIResult] = []
        for payload in payloads:
            outputs.append(self.engine.process_payload(payload))
        self.logger.info("sii_batch_processed", extra={"frames": len(outputs)})
        return outputs

    def run_input_file(self, input_path: str | Path) -> list[SIIResult]:
        path = Path(input_path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            payloads = load_frames_from_json(str(path))
        elif suffix == ".csv":
            payloads = load_frames_from_csv(str(path))
        else:
            raise SIIValidationError(f"Unsupported input format: {suffix!r}")
        return self.run_payloads(payloads)

    def write_output_file(self, output_path: str | Path, results: list[SIIResult]) -> None:
        path = Path(output_path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            write_json_report(path, results)
        elif suffix == ".csv":
            write_csv_report(path, results)
        else:
            raise SIIValidationError(f"Unsupported output format: {suffix!r}")
        self.logger.info("sii_report_emitted", extra={"path": str(path), "rows": len(results)})

