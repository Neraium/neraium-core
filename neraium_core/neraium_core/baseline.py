from pydantic import BaseModel, validator
from typing import List


class Baseline(BaseModel):
    baseline_id: str
    system_id: str
    schema_version: str
    signal_order: List[str]
    mean_vector: List[float]
    covariance_matrix: List[List[float]]
    sample_count: int

    @validator("mean_vector")
    def validate_mean_length(cls, v, values):
        if "signal_order" in values and len(v) != len(values["signal_order"]):
            raise ValueError("mean_vector length must match signal_order")
        return v

    @validator("covariance_matrix")
    def validate_covariance(cls, v, values):
        n = len(values.get("signal_order", []))
        if len(v) != n:
            raise ValueError("covariance_matrix must match signal_order size")
        for row in v:
            if len(row) != n:
                raise ValueError("covariance_matrix must be square")
        return v