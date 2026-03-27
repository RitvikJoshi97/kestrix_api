"""Shared pytest fixtures."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from thermal_palette.alignment import IndexLoader
from thermal_palette.models import GroupBy, ImageRecord, PaletteType, TemperatureRange


# ---------------------------------------------------------------------------
# Sample records covering multiple properties, azimuths, and flight IDs
# ---------------------------------------------------------------------------

_BASE_DT = datetime(2025, 3, 25, 12, 0, 0, tzinfo=timezone.utc)

SAMPLE_RECORDS: list[ImageRecord] = [
    ImageRecord(
        image_id="img-001",
        filename="img-001.jpg",
        captured_at=_BASE_DT,
        position_lat=51.5,
        position_lng=-1.3,
        azimuth=2.0,        # N
        elevation=45.0,
        camera_pitch=-42.0,
        absolute_altitude=160.0,
        flight_id="flight-aaa",
        property_id="prop-111",
    ),
    ImageRecord(
        image_id="img-002",
        filename="img-002.jpg",
        captured_at=_BASE_DT,
        position_lat=51.5,
        position_lng=-1.3,
        azimuth=88.5,       # E
        elevation=45.0,
        camera_pitch=-42.0,
        absolute_altitude=160.0,
        flight_id="flight-aaa",
        property_id="prop-111",
    ),
    ImageRecord(
        image_id="img-003",
        filename="img-003.jpg",
        captured_at=_BASE_DT,
        position_lat=51.6,
        position_lng=-1.4,
        azimuth=-91.2,      # W
        elevation=45.0,
        camera_pitch=-42.0,
        absolute_altitude=160.0,
        flight_id=None,
        property_id="prop-222",
    ),
    ImageRecord(
        image_id="img-004",
        filename="img-004.jpg",
        captured_at=_BASE_DT,
        position_lat=51.6,
        position_lng=-1.4,
        azimuth=178.0,      # S
        elevation=45.0,
        camera_pitch=-42.0,
        absolute_altitude=160.0,
        flight_id=None,
        property_id="prop-222",
    ),
    ImageRecord(
        image_id="img-005",
        filename="img-005.jpg",
        captured_at=_BASE_DT,
        position_lat=51.7,
        position_lng=-1.5,
        azimuth=45.0,       # NE (exactly on boundary → NE)
        elevation=45.0,
        camera_pitch=-42.0,
        absolute_altitude=160.0,
        flight_id="flight-bbb",
        property_id="prop-333",
    ),
]


@pytest.fixture
def sample_records() -> list[ImageRecord]:
    return SAMPLE_RECORDS.copy()


@pytest.fixture
def random_temp_array() -> np.ndarray:
    """10×10 float32 array with known distribution (0–40°C plus two outliers)."""
    rng = np.random.default_rng(42)
    arr = rng.uniform(10.0, 35.0, (10, 10)).astype(np.float32)
    arr[0, 0] = 0.0    # cold outlier
    arr[9, 9] = 100.0  # hot outlier
    return arr


@pytest.fixture
def flat_temp_array() -> np.ndarray:
    """All-same-value array for degenerate range tests."""
    return np.full((5, 5), 20.0, dtype=np.float32)


@pytest.fixture
def sample_temp_range() -> TemperatureRange:
    return TemperatureRange(
        min_temp=10.0,
        max_temp=35.0,
        low_percentile=2.0,
        high_percentile=98.0,
        clamped=True,
    )


# ---------------------------------------------------------------------------
# FastAPI test client with mocked IndexLoader
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """TestClient with a mocked IndexLoader backed by SAMPLE_RECORDS."""
    from thermal_palette.api import app

    mock_index = IndexLoader.__new__(IndexLoader)
    mock_index._records = SAMPLE_RECORDS.copy()

    with patch.object(app, "state") as mock_state:
        mock_state.index = mock_index
        with TestClient(app) as c:
            yield c


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "outputs"
    out.mkdir()
    return out
