"""Data source and output sink adapters for agnostic runner."""

from .base import DataSourceAdapter, OutputSink, Frame, Result, RunContext
from .csv_source import CsvDataSource
from .json_sink import JsonFileSink, JsonLineSink
from .kafka_source import KafkaDataSource
from .kafka_sink import KafkaSink
from .api_source import ApiStreamSource
from .database_sink import DatabaseSink
from .webhook_sink import WebhookSink

__all__ = [
    "DataSourceAdapter",
    "OutputSink",
    "Frame",
    "Result",
    "RunContext",
    "CsvDataSource",
    "JsonFileSink",
    "JsonLineSink",
    "KafkaDataSource",
    "KafkaSink",
    "ApiStreamSource",
    "DatabaseSink",
    "WebhookSink",
]
