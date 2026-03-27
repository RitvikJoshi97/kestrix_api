"""Tests for alignment.py — I/O orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thermal_palette.alignment import align_group, load_temp_array
from thermal_palette.models import AlignmentSpec, GroupBy, PaletteType


def _write_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


class TestLoadTempArray:
    def test_loads_array(self, tmp_path, sample_records):
        record = sample_records[0]
        stem = Path(record.filename).stem
        arr = np.full((10, 10), 20.0, dtype=np.float32)
        _write_npy(tmp_path / f"{stem}.npy", arr)

        loaded = load_temp_array(record, tmp_path)
        assert loaded.shape == (10, 10)
        assert loaded.dtype == np.float32

    def test_raises_for_missing_file(self, tmp_path, sample_records):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_temp_array(sample_records[0], tmp_path)


class TestAlignGroup:
    @pytest.fixture
    def temps_dir(self, tmp_path, sample_records):
        """Write a small .npy for each sample record."""
        td = tmp_path / "thermal_temps"
        for i, r in enumerate(sample_records):
            stem = Path(r.filename).stem
            # Each array has a slightly different mean so ranges vary
            arr = np.full((8, 8), 10.0 + i * 5, dtype=np.float32)
            arr[0, 0] = 0.0    # cold outlier
            arr[7, 7] = 100.0  # hot outlier
            _write_npy(td / f"{stem}.npy", arr)
        return td

    def test_align_group_produces_output_files(
        self, tmp_path, sample_records, temps_dir, tmp_output_dir
    ):
        spec = AlignmentSpec(
            group_by=GroupBy.PROPERTY,
            group_key="prop-111",
            palette=PaletteType.IRON,
            low_percentile=2.0,
            high_percentile=98.0,
        )
        prop_111_records = [r for r in sample_records if r.property_id == "prop-111"]
        result = align_group(prop_111_records, spec, temps_dir, tmp_output_dir)

        assert result.rendered_count == len(prop_111_records)
        for image_id, path in result.output_paths.items():
            assert Path(path).exists()

    def test_shared_temperature_range_in_result(
        self, tmp_path, sample_records, temps_dir, tmp_output_dir
    ):
        spec = AlignmentSpec(
            group_by=GroupBy.AZIMUTH,
            group_key="N",
            palette=PaletteType.WHITE_HOT,
            low_percentile=0.0,
            high_percentile=100.0,
        )
        n_records = [r for r in sample_records if r.image_id == "img-001"]
        result = align_group(n_records, spec, temps_dir, tmp_output_dir)

        assert result.temperature_range.min_temp < result.temperature_range.max_temp

    def test_single_image_group(self, tmp_path, sample_records, temps_dir, tmp_output_dir):
        spec = AlignmentSpec(
            group_by=GroupBy.PROPERTY,
            group_key="prop-333",
            palette=PaletteType.ARCTIC,
            low_percentile=2.0,
            high_percentile=98.0,
        )
        prop_333_records = [r for r in sample_records if r.property_id == "prop-333"]
        result = align_group(prop_333_records, spec, temps_dir, tmp_output_dir)
        assert result.rendered_count == 1

    def test_output_path_structure(self, sample_records, temps_dir, tmp_output_dir):
        spec = AlignmentSpec(
            group_by=GroupBy.PROPERTY,
            group_key="prop-111",
            palette=PaletteType.IRON,
        )
        prop_111_records = [r for r in sample_records if r.property_id == "prop-111"]
        result = align_group(prop_111_records, spec, temps_dir, tmp_output_dir)

        for path_str in result.output_paths.values():
            p = Path(path_str)
            # outputs/{group_by}/{group_key}/{image_id}.jpg
            assert p.parts[-3] == "property"
            assert p.parts[-2] == "prop-111"
            assert p.suffix == ".jpg"
