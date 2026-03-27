from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CompassDirection(str, Enum):
    N = "N"
    NE = "NE"
    E = "E"
    SE = "SE"
    S = "S"
    SW = "SW"
    W = "W"
    NW = "NW"


class PaletteType(str, Enum):
    WHITE_HOT = "WHITE_HOT"
    IRON = "IRON"
    RAINBOW_HC = "RAINBOW_HC"
    ARCTIC = "ARCTIC"


class GroupBy(str, Enum):
    PROPERTY = "property"
    FLIGHT = "flight"
    AZIMUTH = "azimuth"
    DATE = "date"
    PROPERTY_AZIMUTH = "property_azimuth"
    PROPERTY_FLIGHT = "property_flight"


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


class ImageRecord(BaseModel):
    image_id: str
    filename: str
    captured_at: datetime
    position_lat: float
    position_lng: float
    azimuth: float
    elevation: float
    camera_pitch: float
    absolute_altitude: float
    flight_id: Optional[str] = None
    property_id: str


class ImageGroup(BaseModel):
    group_key: str
    group_by: GroupBy
    image_count: int
    image_ids: list[str]


class TemperatureRange(BaseModel):
    min_temp: float
    max_temp: float
    low_percentile: float
    high_percentile: float
    clamped: bool


class AlignmentSpec(BaseModel):
    group_by: GroupBy
    group_key: str
    palette: PaletteType = PaletteType.IRON
    low_percentile: float = Field(default=2.0, ge=0.0, le=100.0)
    high_percentile: float = Field(default=98.0, ge=0.0, le=100.0)


class AlignmentResult(BaseModel):
    spec: AlignmentSpec
    temperature_range: TemperatureRange
    output_paths: dict[str, str]
    rendered_count: int
    duration_seconds: float


class Job(BaseModel):
    job_id: str
    status: JobStatus
    spec: AlignmentSpec
    result: Optional[AlignmentResult] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------


class AlignRequest(BaseModel):
    group_by: GroupBy
    group_key: str
    palette: PaletteType = PaletteType.IRON
    low_percentile: float = Field(default=2.0, ge=0.0, le=100.0)
    high_percentile: float = Field(default=98.0, ge=0.0, le=100.0)


class AlignResponse(BaseModel):
    job_id: str
    status: JobStatus
    spec: AlignmentSpec
    created_at: datetime


class ImageRecordDetail(BaseModel):
    """ImageRecord with derived compass_direction field for API responses."""

    image_id: str
    filename: str
    captured_at: datetime
    azimuth: float
    compass_direction: CompassDirection
    flight_id: Optional[str] = None
    property_id: str


class GroupDetailResponse(BaseModel):
    group_key: str
    group_by: GroupBy
    image_count: int
    images: list[ImageRecordDetail]


class GroupListResponse(BaseModel):
    group_by: GroupBy
    total_groups: int
    groups: list[ImageGroup]


class HealthResponse(BaseModel):
    status: str
    image_count: int
    version: str


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    spec: AlignmentSpec
    result: Optional[AlignmentResult] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
