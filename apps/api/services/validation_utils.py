from __future__ import annotations


def actionable_validation_detail(message: str) -> str:
    text = str(message or "").strip()
    if not text:
        return "Validation failed. Check required fields and payload structure."
    if text.startswith("Invalid CSV row"):
        if "Invalid timestamp" in text:
            return f"{text} Use ISO-8601 timestamps like 2026-01-01T00:00:00+00:00."
        if "Invalid signal value" in text or "Invalid signal type" in text:
            return f"{text} Ensure all sensor values are numeric or blank."
        return text
    if "Invalid timestamp" in text:
        return "Invalid timestamp format. Use ISO-8601 timestamps like 2026-01-01T00:00:00+00:00."
    if "CSV must include" in text or "missing required columns" in text:
        return f"{text} Ensure CSV header includes timestamp, site_id, asset_id plus at least one sensor column."
    if "Could not infer" in text or "Provide a column_mapping" in text:
        return (
            f"{text} Use POST /ingest/csv/preview with a sample of your file, then send column_mapping "
            "(timestamp, asset_id, optional site_id, sensor_columns) with ingest."
        )
    if "Mapping requires" in text or "not present in the CSV header" in text:
        return (
            f"{text} Open the upload mapping panel and assign time, asset/entity, optional site, and one or more "
            "numeric sensor columns."
        )
    if "Invalid signal value" in text or "Invalid signal type" in text:
        return f"{text} Ensure all sensor values are numeric or blank."
    return text
