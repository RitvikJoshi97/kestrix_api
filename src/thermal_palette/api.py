"""FastAPI application — thermal palette alignment service."""

from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from . import __version__
from .alignment import IndexLoader, align_group_async, job_store, load_temp_array
from .config import settings
from .grouping import azimuth_to_compass, get_groups, resolve_group
from .models import (
    AlignRequest,
    AlignResponse,
    AlignmentSpec,
    GroupBy,
    GroupDetailResponse,
    GroupListResponse,
    HealthResponse,
    ImageRecordDetail,
    JobResponse,
    JobStatus,
    PaletteType,
    TemperatureRange,
)
from .palette import compute_percentile_range, render_image

# Shared temperature range cache — keyed by (group_by, group_key, low_pct, high_pct).
# Palette does not affect the range, only the colour mapping.
_range_cache: dict[tuple, TemperatureRange] = {}


# ---------------------------------------------------------------------------
# Application lifespan (replaces deprecated on_event)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(application: FastAPI):
    application.state.index = IndexLoader(settings.index_csv)
    yield


app = FastAPI(
    title="Thermal Palette Alignment Service",
    version=__version__,
    description=(
        "Aligns thermal image colour palettes so that the same temperature "
        "is rendered with the same colour across any logical group of images."
    ),
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Render-Ms", "X-Cache"],
)


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


def get_index() -> IndexLoader:
    return app.state.index  # type: ignore[attr-defined]


Index = Annotated[IndexLoader, Depends(get_index)]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/healthz", response_model=HealthResponse, tags=["meta"])
def healthz(index: Index) -> HealthResponse:
    return HealthResponse(status="ok", image_count=index.count, version=__version__)


@app.get("/groups", response_model=GroupListResponse, tags=["groups"])
def list_groups(
    index: Index,
    group_by: GroupBy = Query(..., description="Dimension to group images by"),
) -> GroupListResponse:
    groups = get_groups(index.get_all(), group_by)
    return GroupListResponse(
        group_by=group_by,
        total_groups=len(groups),
        groups=groups,
    )


@app.get("/groups/{group_by}/{group_key}", response_model=GroupDetailResponse, tags=["groups"])
def get_group_detail(
    group_by: GroupBy,
    group_key: str,
    index: Index,
) -> GroupDetailResponse:
    records = resolve_group(index.get_all(), group_by, group_key)
    if not records:
        raise HTTPException(status_code=404, detail=f"Group '{group_key}' not found for {group_by}")
    images = [
        ImageRecordDetail(
            image_id=r.image_id,
            filename=r.filename,
            captured_at=r.captured_at,
            azimuth=r.azimuth,
            compass_direction=azimuth_to_compass(r.azimuth),
            flight_id=r.flight_id,
            property_id=r.property_id,
        )
        for r in records
    ]
    return GroupDetailResponse(
        group_key=group_key,
        group_by=group_by,
        image_count=len(images),
        images=images,
    )


@app.post("/align", response_model=AlignResponse, status_code=202, tags=["alignment"])
async def submit_alignment(
    body: AlignRequest,
    background_tasks: BackgroundTasks,
    index: Index,
) -> AlignResponse:
    """Submit an alignment job. Returns immediately with a job_id to poll."""
    records = resolve_group(index.get_all(), body.group_by, body.group_key)
    if not records:
        raise HTTPException(
            status_code=404,
            detail=f"Group '{body.group_key}' not found for {body.group_by}",
        )

    spec = AlignmentSpec(
        group_by=body.group_by,
        group_key=body.group_key,
        palette=body.palette,
        low_percentile=body.low_percentile,
        high_percentile=body.high_percentile,
    )
    job = job_store.create(spec)

    async def _run() -> None:
        job_store.update_running(job.job_id)
        try:
            result = await align_group_async(
                records, spec, settings.temps_dir, settings.output_dir
            )
            job_store.update_result(job.job_id, result)
        except Exception as exc:  # noqa: BLE001
            job_store.update_error(job.job_id, str(exc))

    background_tasks.add_task(_run)

    return AlignResponse(
        job_id=job.job_id,
        status=JobStatus.PENDING,
        spec=spec,
        created_at=job.created_at,
    )


@app.get("/jobs/{job_id}", response_model=JobResponse, tags=["alignment"])
def get_job(job_id: str) -> JobResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        spec=job.spec,
        result=job.result,
        error=job.error,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@app.get("/images/{image_id}", tags=["images"])
def serve_original(image_id: str, index: Index):
    """Serve the original (unaligned) image file."""
    record = next((r for r in index.get_all() if r.image_id == image_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Image '{image_id}' not found")
    img_path = settings.images_dir / record.filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image file missing on disk")
    return FileResponse(img_path, media_type="image/jpeg")


@app.get("/render/{group_by}/{group_key}/{image_id}", tags=["images"])
def render_aligned(
    group_by: GroupBy,
    group_key: str,
    image_id: str,
    index: Index,
    palette: PaletteType = Query(PaletteType.IRON),
    low: float = Query(2.0, ge=0.0, le=100.0),
    high: float = Query(98.0, ge=0.0, le=100.0),
):
    """Render a single image with the group-level shared temperature scale.

    The shared scale is computed once per (group, low, high) combination and
    cached in memory. Palette is applied on top — changing palette is free.
    """
    records = resolve_group(index.get_all(), group_by, group_key)
    if not records:
        raise HTTPException(status_code=404, detail=f"Group '{group_key}' not found for {group_by}")

    record = next((r for r in records if r.image_id == image_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Image '{image_id}' not in group '{group_key}'")

    t_start = time.perf_counter()

    cache_key = (group_by, group_key, low, high)
    if cache_key not in _range_cache:
        arrays = [load_temp_array(r, settings.temps_dir) for r in records]
        _range_cache[cache_key] = compute_percentile_range(arrays, low, high)
        cache_status = "MISS"
    else:
        cache_status = "HIT"

    temp_range = _range_cache[cache_key]
    arr = load_temp_array(record, settings.temps_dir)
    img = render_image(arr, temp_range, palette)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)

    render_ms = round((time.perf_counter() - t_start) * 1000)

    return StreamingResponse(
        buf,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "X-Render-Ms": str(render_ms),
            "X-Cache": cache_status,
        },
    )
