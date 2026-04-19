"""Command-line interface for agnostic runner."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import click
import yaml

from .adapters.registry import get_registry
from .config import EngineConfig
from .runner import AgnosticRunner

logger = logging.getLogger(__name__)


@click.group()
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
@click.pass_context
def cli(ctx: click.Context, log_level: str) -> None:
    """Neraium agnostic runner: unified interface for processing data through structural intelligence engine."""
    logging.basicConfig(level=getattr(logging, log_level))
    ctx.ensure_object(dict)


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), required=True, help="Configuration file (YAML or JSON)")
@click.option("--run-id", default=None, help="Unique run identifier (auto-generated if not provided)")
@click.option("--error-handler", type=click.Choice(["log", "raise", "skip"]), default="log", help="Error handling strategy")
@click.option("--output-metrics", type=click.Path(), default=None, help="Write run metrics to file")
def run(config: str, run_id: str | None, error_handler: str, output_metrics: str | None) -> None:
    """Execute runner with configuration file.

    Example YAML config:
    \b
    data_source:
      type: csv
      config:
        path: data.csv
        timestamp_col: timestamp
        unit_id_col: unit_id
        sensor_prefix: sensor_

    output_sink:
      type: json
      config:
        path: results.json

    engine:
      mode: production
      enable_trajectory_memory: true
      enable_advisory_layers: true
    """
    try:
        # Load configuration
        config_path = Path(config)
        if config_path.suffix in {".yaml", ".yml"}:
            with open(config_path) as f:
                config_data = yaml.safe_load(f)
        else:  # Assume JSON
            with open(config_path) as f:
                config_data = json.load(f)

        # Parse data source
        source_config = config_data.get("data_source", {})
        source_type = source_config.get("type")
        if not source_type:
            raise ValueError("Configuration must include 'data_source.type'")

        registry = get_registry()
        data_source = registry.create_source(source_type, source_config.get("config", {}))
        logger.info(f"Created data source: {source_type}")

        # Parse output sink
        sink_config = config_data.get("output_sink", {})
        sink_type = sink_config.get("type")
        if not sink_type:
            raise ValueError("Configuration must include 'output_sink.type'")

        output_sink = registry.create_sink(sink_type, sink_config.get("config", {}))
        logger.info(f"Created output sink: {sink_type}")

        # Parse engine config
        engine_data = config_data.get("engine", {})
        if isinstance(engine_data, str):
            # Support preset names
            if engine_data == "production":
                engine_config = EngineConfig.production()
            elif engine_data == "research":
                engine_config = EngineConfig.research()
            elif engine_data == "experimental":
                engine_config = EngineConfig.experimental()
            elif engine_data == "minimal":
                engine_config = EngineConfig.minimal()
            else:
                raise ValueError(f"Unknown engine preset: {engine_data}")
        else:
            engine_config = EngineConfig.from_dict(engine_data)
        logger.info(f"Engine mode: {engine_config.mode}")

        # Create and run
        runner = AgnosticRunner(
            data_source=data_source,
            output_sink=output_sink,
            engine_config=engine_config,
            run_id=run_id,
            error_handler=error_handler,
        )
        metrics = runner.run()

        # Output results
        click.echo(f"\n{'=' * 60}")
        click.echo(f"Run {metrics.run_id}: {metrics.status.upper()}")
        click.echo(f"{'=' * 60}")
        click.echo(f"Frames processed: {metrics.frame_count}")
        click.echo(f"Errors: {metrics.error_count}")
        click.echo(f"Duration: {metrics.elapsed_seconds:.2f}s")
        click.echo(f"Throughput: {metrics.frames_per_second:.2f} frames/sec")
        click.echo(f"Engine: {metrics.engine_mode}")
        click.echo(f"Source: {metrics.source_type}")
        click.echo(f"Sink: {metrics.sink_type}")
        if metrics.summary:
            click.echo(f"Summary: {json.dumps(metrics.summary, indent=2, default=str)}")

        # Write metrics if requested
        if output_metrics:
            output_path = Path(output_metrics)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(metrics.to_dict(), f, indent=2, default=str)
            click.echo(f"\nMetrics saved to: {output_path}")

        sys.exit(0 if metrics.status == "success" else 1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Runner failed")
        sys.exit(1)


@cli.command()
@click.argument("source_type", required=False)
def list_adapters(source_type: str | None) -> None:
    """List available data sources and output sinks."""
    registry = get_registry()

    click.echo("=" * 60)
    click.echo("DATA SOURCES")
    click.echo("=" * 60)
    for name in registry.list_sources():
        click.echo(f"  {name}")

    click.echo("\n" + "=" * 60)
    click.echo("OUTPUT SINKS")
    click.echo("=" * 60)
    for name in registry.list_sinks():
        click.echo(f"  {name}")

    click.echo("\n" + "=" * 60)
    click.echo("ENGINE PRESETS")
    click.echo("=" * 60)
    click.echo("  production  (optimized for deployment)")
    click.echo("  research    (full analysis capabilities)")
    click.echo("  experimental (all components enabled)")
    click.echo("  minimal     (core only)")


@cli.command()
@click.option("--template", type=click.Choice(["csv-to-json", "api-to-kafka", "full-featured"]), default="csv-to-json")
@click.option("--output", "-o", type=click.Path(), required=True, help="Output configuration file")
def generate_config(template: str, output: str) -> None:
    """Generate example configuration file.

    Templates:
    \b
    - csv-to-json: Read CSV, output JSON
    - api-to-kafka: Poll API, send to Kafka
    - full-featured: All components and options
    """
    if template == "csv-to-json":
        config = {
            "data_source": {
                "type": "csv",
                "config": {
                    "path": "data.csv",
                    "timestamp_col": "timestamp",
                    "unit_id_col": "unit_id",
                    "sensor_prefix": "sensor_",
                },
            },
            "output_sink": {
                "type": "json",
                "config": {
                    "path": "results.json",
                },
            },
            "engine": "production",
        }
    elif template == "api-to-kafka":
        config = {
            "data_source": {
                "type": "api",
                "config": {
                    "url": "http://localhost:8000/frames",
                    "poll_interval_ms": 1000,
                    "max_polls": 100,
                    "timeout_sec": 10,
                },
            },
            "output_sink": {
                "type": "kafka",
                "config": {
                    "brokers": ["localhost:9092"],
                    "topic": "neraium-results",
                },
            },
            "engine": "production",
        }
    else:  # full-featured
        config = {
            "data_source": {
                "type": "csv",
                "config": {
                    "path": "data.csv",
                    "timestamp_col": "timestamp",
                    "unit_id_col": "unit_id",
                    "sensor_prefix": "sensor_",
                    "skip_rows": 0,
                },
            },
            "output_sink": {
                "type": "jsonl",
                "config": {
                    "path": "results.jsonl",
                },
            },
            "engine": {
                "mode": "production",
                "enable_trajectory_memory": True,
                "enable_law_extraction": False,
                "enable_law_engine": False,
                "enable_intervention_memory": True,
                "enable_reliability_calibration": True,
                "enable_cross_system_transfer": False,
                "enable_advisory_layers": True,
                "enable_experimental_layers": False,
            },
        }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        if output_path.suffix in {".yaml", ".yml"}:
            yaml.dump(config, f, default_flow_style=False)
        else:
            json.dump(config, f, indent=2)

    click.echo(f"Configuration template saved to: {output_path}")


if __name__ == "__main__":
    cli(obj={})
