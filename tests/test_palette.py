"""Tests for palette.py — pure rendering functions."""

from __future__ import annotations

import numpy as np
import pytest

from thermal_palette.models import PaletteType, TemperatureRange
from thermal_palette.palette import (
    compute_percentile_range,
    convert_to_palette,
    get_colour_stops,
    render_image,
)


class TestGetColourStops:
    def test_all_palettes_return_array(self):
        for palette in PaletteType:
            stops = get_colour_stops(palette)
            assert stops.ndim == 2
            assert stops.shape[1] == 3

    def test_arctic_palette_exists(self):
        """ARCTIC must not raise ValueError — it was missing in the prototype."""
        stops = get_colour_stops(PaletteType.ARCTIC)
        assert len(stops) >= 2

    def test_colour_values_in_range(self):
        for palette in PaletteType:
            stops = get_colour_stops(palette)
            assert stops.min() >= 0
            assert stops.max() <= 255


class TestConvertToPalette:
    def test_output_shape(self, random_temp_array, sample_temp_range):
        rgb = convert_to_palette(random_temp_array, sample_temp_range, PaletteType.IRON)
        h, w = random_temp_array.shape
        assert rgb.shape == (h, w, 3)

    def test_output_dtype_uint8(self, random_temp_array, sample_temp_range):
        rgb = convert_to_palette(random_temp_array, sample_temp_range, PaletteType.IRON)
        assert rgb.dtype == np.uint8

    def test_degenerate_range_no_nan(self, flat_temp_array):
        """min_temp == max_temp must not produce NaN or divide-by-zero."""
        temp_range = TemperatureRange(
            min_temp=20.0, max_temp=20.0,
            low_percentile=0.0, high_percentile=100.0, clamped=False
        )
        rgb = convert_to_palette(flat_temp_array, temp_range, PaletteType.WHITE_HOT)
        assert not np.isnan(rgb.astype(float)).any()
        assert rgb.shape == (*flat_temp_array.shape, 3)

    def test_all_palettes_render_without_error(self, random_temp_array, sample_temp_range):
        for palette in PaletteType:
            rgb = convert_to_palette(random_temp_array, sample_temp_range, palette)
            assert rgb.shape[2] == 3

    def test_hot_pixels_map_to_hot_colour(self, sample_temp_range):
        """A pixel at max_temp should map to the last colour stop (white for WHITE_HOT)."""
        arr = np.array([[sample_temp_range.max_temp]], dtype=np.float32)
        rgb = convert_to_palette(arr, sample_temp_range, PaletteType.WHITE_HOT)
        assert rgb[0, 0, 0] == 255  # white
        assert rgb[0, 0, 1] == 255
        assert rgb[0, 0, 2] == 255

    def test_cold_pixels_map_to_cold_colour(self, sample_temp_range):
        """A pixel at min_temp should map to the first colour stop (black for WHITE_HOT)."""
        arr = np.array([[sample_temp_range.min_temp]], dtype=np.float32)
        rgb = convert_to_palette(arr, sample_temp_range, PaletteType.WHITE_HOT)
        assert rgb[0, 0, 0] == 0   # black
        assert rgb[0, 0, 1] == 0
        assert rgb[0, 0, 2] == 0


class TestComputePercentileRange:
    def test_returns_temperature_range(self, random_temp_array):
        result = compute_percentile_range([random_temp_array], 2.0, 98.0)
        assert result.min_temp < result.max_temp

    def test_percentile_clamps_outliers(self, random_temp_array):
        """With outliers at 0 and 100, p2/p98 range should be well within 0–100."""
        result = compute_percentile_range([random_temp_array], 2.0, 98.0)
        # The bulk of the array is in [10, 35]; outliers should not dominate
        assert result.min_temp > 0.0
        assert result.max_temp < 100.0

    def test_clamped_flag_set_when_percentiles_not_extremes(self, random_temp_array):
        result = compute_percentile_range([random_temp_array], 2.0, 98.0)
        assert result.clamped is True

    def test_clamped_flag_false_for_full_range(self, random_temp_array):
        result = compute_percentile_range([random_temp_array], 0.0, 100.0)
        assert result.clamped is False

    def test_multiple_arrays_share_range(self):
        arr1 = np.full((5, 5), 15.0, dtype=np.float32)
        arr2 = np.full((5, 5), 30.0, dtype=np.float32)
        result = compute_percentile_range([arr1, arr2], 0.0, 100.0)
        assert result.min_temp == pytest.approx(15.0, abs=0.1)
        assert result.max_temp == pytest.approx(30.0, abs=0.1)

    def test_degenerate_all_same_value(self, flat_temp_array):
        """All pixels the same temperature — range must be expanded, not zero."""
        result = compute_percentile_range([flat_temp_array], 2.0, 98.0)
        assert result.max_temp > result.min_temp


class TestRenderImage:
    def test_returns_pil_image(self, random_temp_array, sample_temp_range):
        from PIL import Image

        img = render_image(random_temp_array, sample_temp_range, PaletteType.IRON)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (random_temp_array.shape[1], random_temp_array.shape[0])
