"""Typer CLI — local demo entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .alignment import IndexLoader, align_group, job_store
from .config import settings
from .grouping import get_groups, resolve_group
from .models import AlignmentSpec, GroupBy, PaletteType

app = typer.Typer(
    name="thermal-palette",
    help="Thermal image palette alignment — CLI demo.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def groups(
    by: GroupBy = typer.Option(..., "--by", help="Grouping dimension"),
    min_size: int = typer.Option(1, "--min-size", help="Only show groups with at least N images"),
) -> None:
    """List all image groups for the given grouping dimension."""
    index = IndexLoader(settings.index_csv)
    all_groups = get_groups(index.get_all(), by)
    filtered = [g for g in all_groups if g.image_count >= min_size]

    table = Table(title=f"Groups by {by.value} (≥{min_size} images)")
    table.add_column("Group key", style="cyan", no_wrap=True)
    table.add_column("Images", justify="right")

    for g in filtered:
        table.add_row(g.group_key, str(g.image_count))

    console.print(table)
    console.print(f"\n[bold]{len(filtered)}[/bold] groups total")


@app.command()
def align(
    by: GroupBy = typer.Option(..., "--by", help="Grouping dimension"),
    key: str = typer.Option(..., "--key", help="Group key (e.g. property UUID or 'N')"),
    palette: PaletteType = typer.Option(PaletteType.IRON, "--palette", help="Colour palette"),
    output: Path = typer.Option(settings.output_dir, "--output", help="Output directory"),
    low_pct: float = typer.Option(
        settings.default_low_percentile, "--low-pct", help="Lower percentile clamp (0-100)"
    ),
    high_pct: float = typer.Option(
        settings.default_high_percentile, "--high-pct", help="Upper percentile clamp (0-100)"
    ),
) -> None:
    """Align and re-render all images in a group with a shared temperature range."""
    index = IndexLoader(settings.index_csv)
    records = resolve_group(index.get_all(), by, key)

    if not records:
        console.print(f"[red]No images found for {by.value}={key!r}[/red]")
        raise typer.Exit(1)

    console.print(f"Aligning [bold]{len(records)}[/bold] images ({by.value}={key!r})…")

    spec = AlignmentSpec(
        group_by=by,
        group_key=key,
        palette=palette,
        low_percentile=low_pct,
        high_percentile=high_pct,
    )
    result = align_group(records, spec, settings.temps_dir, output)

    console.print(f"[green]Done.[/green] Rendered {result.rendered_count} images in {result.duration_seconds:.2f}s")
    console.print(
        f"Shared range: [bold]{result.temperature_range.min_temp:.2f}°[/bold] – "
        f"[bold]{result.temperature_range.max_temp:.2f}°[/bold] "
        f"(p{result.temperature_range.low_percentile}–p{result.temperature_range.high_percentile})"
    )
    console.print(f"Output: {output / by.value / key}")


@app.command()
def validate(
    data: Optional[Path] = typer.Option(None, "--data", help="Override data directory"),
) -> None:
    """Check that every row in image_index.csv has a corresponding .npy file."""
    data_dir = data or settings.data_dir
    csv_path = data_dir / "image_index.csv"
    temps_dir = data_dir / "thermal_temps"

    index = IndexLoader(csv_path)
    records = index.get_all()

    missing = []
    for r in records:
        stem = Path(r.filename).stem
        npy = temps_dir / f"{stem}.npy"
        if not npy.exists():
            missing.append((r.image_id, str(npy)))

    if missing:
        console.print(f"[red]{len(missing)} missing .npy files:[/red]")
        for image_id, path in missing[:20]:
            console.print(f"  {image_id}  →  {path}")
        if len(missing) > 20:
            console.print(f"  … and {len(missing) - 20} more")
        raise typer.Exit(1)
    else:
        console.print(f"[green]All {len(records)} temperature arrays present.[/green]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
) -> None:
    """Launch the FastAPI service with uvicorn."""
    import uvicorn

    uvicorn.run(
        "thermal_palette.api:app",
        host=host,
        port=port,
        reload=reload,
    )
