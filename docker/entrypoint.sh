#!/usr/bin/env sh
set -eu

# Ensure writable parent exists for local persistence.
db_path="${NERAIUM_DB_PATH:-/data/neraium.db}"
db_dir="$(dirname "$db_path")"
mkdir -p "$db_dir"

# Optional integration config path can point to mounted read-only config.
# If unset, app still starts and pull mapping falls back to defaults.

exec python -m apps.api.main
