"""Pure palette rendering functions — no I/O."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .models import PaletteType, TemperatureRange

# ---------------------------------------------------------------------------
# Palette colour stop definitions
# Each palette is a list of [R, G, B] stops that are linearly interpolated.
# Temperature maps to colour from cold (index 0) to hot (index -1).
# ---------------------------------------------------------------------------

_WHITE = [255, 255, 255]
_BLACK = [0, 0, 0]
_YELLOW = [255, 255, 0]
_RED = [200, 0, 0]
_DARK_RED = [100, 0, 0]
_PINK = [219, 0, 170]
_CYAN = [0, 237, 216]
_DARK_GREEN = [0, 70, 10]
_BLUE = [0, 0, 139]
_INDIGO = [40, 0, 160]
_DEEP_NAVY = [0, 0, 60]
_MAGENTA = [180, 0, 180]
_RED_ORANGE = [255, 70, 0]
_DARK_YELLOW = [230, 170, 0]
_LIGHT_BLUE = [173, 216, 230]
_ICY_BLUE = [100, 180, 255]

PALETTE_DEFINITIONS: dict[str, list[list[int]]] = {
    "WHITE_HOT": [_BLACK, _WHITE],
    "IRON": [_BLACK, _DEEP_NAVY, _BLUE, _INDIGO, _MAGENTA, _RED, _RED_ORANGE, _DARK_YELLOW, _WHITE],
    "RAINBOW_HC": [_BLACK, _PINK, _BLUE, _CYAN, _DARK_GREEN, _YELLOW, _DARK_RED, _RED, _WHITE],
    # ARCTIC: cold (dark blue) → icy blue → light blue → white (hot)
    # Good for visualising cool structures like building facades in winter.
    "ARCTIC": [_DEEP_NAVY, _BLUE, _ICY_BLUE, _LIGHT_BLUE, _CYAN, _WHITE],
}


def get_colour_stops(palette: PaletteType) -> np.ndarray:
    """Return the (N, 3) float32 colour stop array for the given palette."""
    stops = PALETTE_DEFINITIONS.get(palette.value)
    if stops is None:
        raise ValueError(f"Unknown palette: {palette!r}. Valid options: {list(PALETTE_DEFINITIONS)}")
    return np.array(stops, dtype=np.float32)


def convert_to_palette(
    temps: np.ndarray,
    temp_range: TemperatureRange,
    palette: PaletteType,
) -> np.ndarray:
    """Convert a 2-D temperature array to an (H, W, 3) uint8 RGB image.

    Uses the shared TemperatureRange so that colours are consistent across
    all images rendered with the same range (palette alignment).
    """
    h, w = temps.shape
    denom = temp_range.max_temp - temp_range.min_temp
    if denom == 0.0:
        # Degenerate range — return the midpoint colour for every pixel.
        stops = get_colour_stops(palette)
        mid_colour = stops[len(stops) // 2].astype(np.uint8)
        return np.broadcast_to(mid_colour, (h, w, 3)).copy()

    norm = np.clip((temps - temp_range.min_temp) / denom, 0.0, 1.0)
    stops = get_colour_stops(palette)
    n = len(stops)

    scaled = norm * (n - 1)
    # Clip i0 to n-2 so that i1 = i0+1 is always a valid index (handles norm == 1.0).
    i0 = np.floor(scaled).astype(np.intp).clip(0, n - 2)
    i1 = i0 + 1
    frac = (scaled - i0)[..., np.newaxis]  # broadcast over RGB channels

    rgb = (1.0 - frac) * stops[i0] + frac * stops[i1]
    return rgb.astype(np.uint8)


def compute_percentile_range(
    arrays: list[np.ndarray],
    low_pct: float = 2.0,
    high_pct: float = 98.0,
) -> TemperatureRange:
    """Compute a shared temperature range across a group of temperature arrays.

    Percentile clamping prevents a single hot pixel (e.g. an HVAC unit at 60 °C)
    from compressing all the colour variation in the rest of the image group.
    """
    combined = np.concatenate([a.ravel() for a in arrays])
    min_temp = float(np.percentile(combined, low_pct))
    max_temp = float(np.percentile(combined, high_pct))

    # Guard: if the range collapses (rare, but possible in synthetic data),
    # expand it slightly so downstream division never produces NaN.
    if min_temp == max_temp:
        min_temp -= 0.5
        max_temp += 0.5

    return TemperatureRange(
        min_temp=min_temp,
        max_temp=max_temp,
        low_percentile=low_pct,
        high_percentile=high_pct,
        clamped=(low_pct > 0.0 or high_pct < 100.0),
    )


def render_image(
    temp_array: np.ndarray,
    temp_range: TemperatureRange,
    palette: PaletteType,
) -> Image.Image:
    """Return a PIL Image rendered with the given palette and temperature range."""
    rgb = convert_to_palette(temp_array, temp_range, palette)
    return Image.fromarray(rgb, mode="RGB")
