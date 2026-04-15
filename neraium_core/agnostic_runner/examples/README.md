# Agnostic Runner Examples

This directory contains example configurations for the Neraium agnostic runner, demonstrating different data source and output sink combinations.

## Quick Start

1. **List available adapters:**
   ```bash
   python -m neraium_core.agnostic_runner.cli list-adapters
   ```

2. **Generate a template configuration:**
   ```bash
   python -m neraium_core.agnostic_runner.cli generate-config --template csv-to-json --output my-config.yaml
   ```

3. **Run with a configuration:**
   ```bash
   python -m neraium_core.agnostic_runner.cli run --config my-config.yaml
   ```

## Example Configurations

### csv-to-json.yaml
**Use case:** Process a batch of sensor data from a CSV file and output results as JSON.

**Best for:**
- Validation and testing
- Batch processing of historical data
- Integration with simple ETL pipelines

**Features:**
- Simple CSV input
- Single JSON file output
- Production-optimized engine

### api-to-kafka.yaml
**Use case:** Poll sensor data from a REST API and stream results to Kafka for distributed processing.

**Best for:**
- Real-time monitoring systems
- Distributed processing pipelines
- Integration with Kafka-based architectures

**Features:**
- Configurable polling interval
- Direct Kafka streaming
- Partitioning by unit_id

### kafka-to-database.yaml
**Use case:** Consume sensor data from Kafka and persist results to a SQL database.

**Best for:**
- Long-term analysis
- Historical querying and reporting
- Multi-database support (SQLite, PostgreSQL, MySQL)

**Features:**
- Full Kafka consumer capabilities
- Automatic table creation
- Batch inserts for performance
- Full research mode (all engine components)

### csv-to-webhook.yaml
**Use case:** Process batch data and send results to HTTP webhooks for downstream actions.

**Best for:**
- Triggering integrations (Slack, PagerDuty, etc.)
- Dashboard updates
- Event-driven architectures

**Features:**
- Configurable batching
- Automatic retry with exponential backoff
- Custom authentication headers

## Configuration Structure

All configurations follow this structure:

```yaml
data_source:
  type: <source_type>        # csv, kafka, api
  config:
    <source_specific_config>

output_sink:
  type: <sink_type>          # json, jsonl, kafka, database, webhook
  config:
    <sink_specific_config>

engine:
  mode: <mode>               # production, research_assistive, experimental, minimal
  enable_*: true/false       # Component-level feature toggles
```

## Data Source Adapters

### csv
Read frames from a CSV file.

**Config:**
- `path` (required): Path to CSV file
- `timestamp_col` (default: "timestamp"): Column name for timestamps
- `unit_id_col` (default: "unit_id"): Column name for unit/equipment IDs
- `sensor_prefix` (default: "sensor_"): Prefix for sensor value columns
- `skip_rows` (default: 0): Rows to skip before header

**CSV Format:**
```
timestamp,unit_id,sensor_temperature,sensor_pressure,sensor_vibration
1704067200.0,pump_001,65.3,101.5,0.12
1704067210.0,pump_001,65.5,101.6,0.13
```

### api
Poll frames from a REST API endpoint.

**Config:**
- `url` (required): API endpoint URL
- `poll_interval_ms` (default: 1000): Polling interval
- `max_polls` (default: null): Max number of polls
- `timeout_sec` (default: 10): Request timeout
- `verify_ssl` (default: true): Verify SSL certificates
- `headers` (default: {}): Custom HTTP headers
- `response_key` (default: "data"): Key in response containing frames

### kafka
Consume frames from a Kafka topic.

**Config:**
- `brokers` (required): List of broker addresses
- `topic` (required): Topic to consume from
- `group_id` (default: "neraium-runner"): Consumer group
- `auto_offset_reset` (default: "earliest"): Where to start
- `max_records` (default: null): Max records to read
- `timeout_ms` (default: 10000): Poll timeout

## Output Sink Adapters

### json
Write results to a single JSON file (array of objects).

**Config:**
- `path` (default: "results.json"): Output file path

### jsonl
Write results to JSONL file (one JSON object per line).

**Config:**
- `path` (default: "results.jsonl"): Output file path

### kafka
Send results to a Kafka topic.

**Config:**
- `brokers` (required): List of broker addresses
- `topic` (required): Topic to produce to
- `key_field` (default: "unit_id"): Field to use as message key

### database
Persist results to a SQL database.

**Config:**
- `connection_string` (required): SQLAlchemy connection string
- `table_name` (default: "neraium_results"): Table name
- `batch_size` (default: 100): Batch insert size
- `json_columns` (default: true): Store data as JSON vs flattened

**Supported databases:**
- SQLite: `sqlite:///path/to/file.db`
- PostgreSQL: `postgresql://user:pass@localhost/dbname`
- MySQL: `mysql+pymysql://user:pass@localhost/dbname`

### webhook
Send results to HTTP webhook endpoints.

**Config:**
- `url` (required): Webhook endpoint URL
- `timeout_sec` (default: 10): Request timeout
- `verify_ssl` (default: true): Verify SSL
- `headers` (default: {}): Custom headers
- `batch_size` (default: 1): Results per webhook call
- `retry_count` (default: 3): Retry attempts
- `retry_delay_ms` (default: 1000): Delay between retries

## Engine Modes

### production (default)
Optimized for deployment with essential features:
- Trajectory memory and intervention tracking
- Reliability calibration
- Advisory layers for operator guidance
- No experimental features

### research_assistive
Full analysis capabilities for research:
- All production features
- Law extraction and law engine
- Cross-system intelligence
- No experimental features

### experimental
All components including experimental:
- All production and research features
- Experimental universal layer
- Falsification intelligence
- Active learning
- Structural sandbox

### minimal
Core engine only:
- Structural analysis only
- No optional components
- Minimal latency
- Lowest memory usage

## Running Examples

```bash
# Generate a config from template
python -m neraium_core.agnostic_runner.cli generate-config \
  --template csv-to-json \
  --output my-config.yaml

# Run with the config
python -m neraium_core.agnostic_runner.cli run \
  --config my-config.yaml \
  --output-metrics metrics.json

# Run with error handling
python -m neraium_core.agnostic_runner.cli run \
  --config my-config.yaml \
  --error-handler skip \
  --log-level DEBUG
```

## Custom Configurations

You can mix and match any data source with any output sink:

```yaml
# Example: CSV input -> multiple outputs
# Just create a config with your chosen combination

data_source:
  type: csv
  config:
    path: data.csv

output_sink:
  type: kafka      # Could be: json, jsonl, kafka, database, webhook
  config:
    brokers: [localhost:9092]
    topic: results
```

## Troubleshooting

1. **"Unknown adapter" error:**
   ```bash
   python -m neraium_core.agnostic_runner.cli list-adapters
   ```
   Check that your adapter type is in the list.

2. **CSV parsing errors:**
   - Ensure column names match your config
   - Check that sensor columns start with the specified prefix
   - Verify timestamp column contains numeric values

3. **Kafka connection errors:**
   - Check broker addresses and ports
   - Verify topic exists
   - Check network connectivity

4. **Database errors:**
   - Verify connection string format
   - Ensure database server is running
   - Check user permissions

## Advanced Usage

### Monitoring a Run

```bash
python -m neraium_core.agnostic_runner.cli run \
  --config config.yaml \
  --run-id my-run-123 \
  --output-metrics metrics.json
```

### Custom Error Handling

- `log` (default): Log errors and continue
- `skip`: Silently skip errors
- `raise`: Stop on first error

```bash
python -m neraium_core.agnostic_runner.cli run \
  --config config.yaml \
  --error-handler raise
```

## Performance Tips

1. **Batch operations:**
   - Set `batch_size` for database and webhook sinks
   - Use JSONL output for large result sets

2. **Sampling:**
   - CSV: Use `skip_rows` for large files
   - API: Use `max_polls` to limit polling
   - Kafka: Set consumer group for parallel processing

3. **Engine tuning:**
   - Use `minimal` mode for speed
   - Use `production` mode for standard deployments
   - Use `research` mode only when needed

## See Also

- [Agnostic Runner API Reference](../../runner.py)
- [Adapter Registry](../../adapters/registry.py)
- [Engine Configuration](../../config.py)
