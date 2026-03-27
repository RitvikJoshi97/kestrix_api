"""Tests for api.py — FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from thermal_palette.alignment import IndexLoader
from thermal_palette.api import app, get_index
from tests.conftest import SAMPLE_RECORDS


@pytest.fixture
def client():
    """TestClient using dependency override — bypasses startup CSV loading."""
    mock_index = IndexLoader.__new__(IndexLoader)
    mock_index._records = SAMPLE_RECORDS.copy()

    app.dependency_overrides[get_index] = lambda: mock_index
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


class TestHealthz:
    def test_returns_200(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200

    def test_image_count(self, client):
        data = client.get("/healthz").json()
        assert data["image_count"] == len(SAMPLE_RECORDS)
        assert data["status"] == "ok"


class TestListGroups:
    def test_group_by_property(self, client):
        resp = client.get("/groups", params={"group_by": "property"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["group_by"] == "property"
        assert data["total_groups"] == 3  # prop-111, prop-222, prop-333

    def test_group_by_azimuth(self, client):
        resp = client.get("/groups", params={"group_by": "azimuth"})
        assert resp.status_code == 200
        data = resp.json()
        keys = {g["group_key"] for g in data["groups"]}
        assert "N" in keys
        assert "E" in keys

    def test_missing_group_by_param_returns_422(self, client):
        resp = client.get("/groups")
        assert resp.status_code == 422

    def test_invalid_group_by_returns_422(self, client):
        resp = client.get("/groups", params={"group_by": "invalid_dimension"})
        assert resp.status_code == 422


class TestGetGroupDetail:
    def test_valid_group(self, client):
        resp = client.get("/groups/property/prop-111")
        assert resp.status_code == 200
        data = resp.json()
        assert data["group_key"] == "prop-111"
        assert data["image_count"] == 2

    def test_images_have_compass_direction(self, client):
        data = client.get("/groups/azimuth/N").json()
        for img in data["images"]:
            assert "compass_direction" in img

    def test_unknown_group_returns_404(self, client):
        resp = client.get("/groups/property/nonexistent-uuid")
        assert resp.status_code == 404


class TestSubmitAlignment:
    def test_returns_202_with_job_id(self, client):
        resp = client.post(
            "/align",
            json={
                "group_by": "property",
                "group_key": "prop-111",
                "palette": "IRON",
                "low_percentile": 2.0,
                "high_percentile": 98.0,
            },
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "PENDING"

    def test_invalid_palette_returns_422(self, client):
        resp = client.post(
            "/align",
            json={
                "group_by": "property",
                "group_key": "prop-111",
                "palette": "NONEXISTENT_PALETTE",
            },
        )
        assert resp.status_code == 422

    def test_unknown_group_returns_404(self, client):
        resp = client.post(
            "/align",
            json={
                "group_by": "property",
                "group_key": "no-such-property",
                "palette": "IRON",
            },
        )
        assert resp.status_code == 404


class TestGetJob:
    def test_unknown_job_returns_404(self, client):
        resp = client.get("/jobs/nonexistent-job-id")
        assert resp.status_code == 404

    def test_pending_job_retrievable(self, client):
        submit_resp = client.post(
            "/align",
            json={
                "group_by": "property",
                "group_key": "prop-111",
                "palette": "IRON",
            },
        )
        job_id = submit_resp.json()["job_id"]

        resp = client.get(f"/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert data["status"] in ("PENDING", "RUNNING", "COMPLETE", "FAILED")
