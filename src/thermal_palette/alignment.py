"""I/O orchestration: load temperature arrays, compute shared range, render, save."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .models import (
    AlignmentResult,
    AlignmentSpec,
    ImageRecord,
    Job,
    JobStatus,
    TemperatureRange,
)
from .palette import compute_percentile_range, render_image


# ---------------------------------------------------------------------------
# Index loader
# ---------------------------------------------------------------------------


class IndexLoader:
    """Loads image_index.csv once and exposes query methods.

    Structured so the underlying storage can be swapped for a DB query layer
    without changing the interface used by alignment.py or api.py.
    """

    def __init__(self, csv_path: Path) -> None:
        df = pd.read_csv(csv_path, parse_dates=["captured_at"])
        records = []
        for _, row in df.iterrows():
            d = row.to_dict()
            # pandas represents missing strings as float NaN; convert to None
            if pd.isna(d.get("flight_id")):
                d["flight_id"] = None
            records.append(ImageRecord(**d))
        self._records = records

    def get_all(self) -> list[ImageRecord]:
        return self._records

    def get_by_ids(self, image_ids: list[str]) -> list[ImageRecord]:
        id_set = set(image_ids)
        return [r for r in self._records if r.image_id in id_set]

    @property
    def count(self) -> int:
        return len(self._records)


# ---------------------------------------------------------------------------
# Per-image I/O helpers
# ---------------------------------------------------------------------------


def load_temp_array(record: ImageRecord, temps_dir: Path) -> np.ndarray:
    """Memory-map the .npy temperature file for a given image record.

    Memory mapping avoids loading the full 508 MB dataset into RAM when
    processing large groups; each array is paged in on demand.
    """
    stem = Path(record.filename).stem
    npy_path = temps_dir / f"{stem}.npy"
    if not npy_path.exists():
        raise FileNotFoundError(f"Temperature array not found: {npy_path}")
    return np.load(npy_path, mmap_mode="r")


# ---------------------------------------------------------------------------
# Core alignment pipeline
# ---------------------------------------------------------------------------


def align_group(
    records: list[ImageRecord],
    spec: AlignmentSpec,
    temps_dir: Path,
    output_dir: Path,
) -> AlignmentResult:
    """Align and render all images in a group with a shared temperature range.

    Steps:
      1. Load temperature arrays (memory-mapped).
      2. Compute a shared percentile-clamped temperature range.
      3. Re-render every image with the shared range.
      4. Save outputs to output_dir/{group_by}/{group_key}/.
    """
    started = time.monotonic()

    # Step 1: load
    arrays: list[np.ndarray] = [load_temp_array(r, temps_dir) for r in records]

    # Step 2: shared range
    temp_range: TemperatureRange = compute_percentile_range(
        arrays, spec.low_percentile, spec.high_percentile
    )

    # Step 3 + 4: render and save
    group_out_dir = output_dir / spec.group_by.value / spec.group_key
    group_out_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, str] = {}
    for record, arr in zip(records, arrays):
        img = render_image(arr, temp_range, spec.palette)
        out_path = group_out_dir / f"{record.image_id}.jpg"
        img.save(out_path, format="JPEG", quality=95)
        output_paths[record.image_id] = str(out_path)

    duration = time.monotonic() - started

    return AlignmentResult(
        spec=spec,
        temperature_range=temp_range,
        output_paths=output_paths,
        rendered_count=len(records),
        duration_seconds=round(duration, 3),
    )


async def align_group_async(
    records: list[ImageRecord],
    spec: AlignmentSpec,
    temps_dir: Path,
    output_dir: Path,
) -> AlignmentResult:
    """Run align_group in a thread-pool executor to avoid blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        align_group,
        records,
        spec,
        temps_dir,
        output_dir,
    )


# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------


class JobStore:
    """Thread-safe in-memory job registry.

    Production replacement: swap the dict for a Redis hash or a DB table.
    The interface (create / get / update_result / update_error) stays identical.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, spec: AlignmentSpec) -> Job:
        job = Job(
            job_id=str(uuid.uuid4()),
            status=JobStatus.PENDING,
            spec=spec,
            created_at=datetime.now(timezone.utc),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def update_running(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            self._jobs[job_id] = job.model_copy(update={"status": JobStatus.RUNNING})

    def update_result(self, job_id: str, result: AlignmentResult) -> None:
        with self._lock:
            job = self._jobs[job_id]
            self._jobs[job_id] = job.model_copy(
                update={
                    "status": JobStatus.COMPLETE,
                    "result": result,
                    "completed_at": datetime.now(timezone.utc),
                }
            )

    def update_error(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            self._jobs[job_id] = job.model_copy(
                update={
                    "status": JobStatus.FAILED,
                    "error": error,
                    "completed_at": datetime.now(timezone.utc),
                }
            )

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._jobs)


# Module-level singleton — shared across the FastAPI app and CLI.
job_store = JobStore()
