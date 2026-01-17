"""
Data contract models for ECG processing service.

These Pydantic models define the input/output JSON formats so teammates can
implement algorithms without changing the interface.
"""
from __future__ import annotations

from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field


class Channel(BaseModel):
    """Channel metadata for a single signal channel."""

    name: str = Field(..., examples=["ECG"])
    unit: str = Field(..., examples=["adc"])
    lead: Optional[str] = Field(default=None, examples=["CH1"])


class EcgSegment(BaseModel):
    """Input payload: a short ECG segment in a unified format."""

    schema_version: Literal["ecg-seg/v1"] = "ecg-seg/v1"
    segment_id: str = Field(..., examples=["demo_1700000000000"])
    sampling_rate_hz: int = Field(..., ge=1, examples=[700])
    start_time_unix_ms: int = Field(..., examples=[1700000000000])
    channels: List[Channel]
    samples: List[List[float]] = Field(
        ...,
        description="2D array with shape [N, C]. Each row is one sample time.",
        examples=[[[30950], [31141], [31279]]],
    )


class BioStatus(BaseModel):
    """Real-time biometric status."""
    timestamp_ms: int
    hr_bpm: float
    hrv_sdnn_ms: float
    signal_quality_score: float  # 0.0 to 1.0
    is_stressed: bool
    notes: Optional[str] = None

class UserBaseline(BaseModel):
    """User physiological baseline by time period."""
    user_id: str
    period_start_hour: int
    avg_hr: float
    avg_sdnn: float
    last_updated: int

class PomodoroSummary(BaseModel):
    """Summary of a completed Pomodoro session."""
    session_id: str
    start_time: int
    end_time: int
    avg_hr: float
    avg_sdnn: float
    stress_percentage: float
    total_samples: int
    plot_url: Optional[str] = None

class EcgFeatures(BaseModel):
    """Output payload: extracted features from ECG segment."""
    schema_version: Literal["ecg-feat/v1"] = "ecg-feat/v1"
    segment_id: str
    p_wave_ms: float
    t_wave_ms: float
    qrs_complex_ms: float
    rr_intervals_ms: List[float]
    hrv_sdnn_ms: float
    hr_bpm: float
    quality: Dict
