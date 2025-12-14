"""Progress display for FDTD simulations.

Provides rich terminal UI for real-time simulation progress tracking including:
- Progress bar with percentage
- Elapsed time and ETA
- Computational throughput (Mcells/s)
- Memory usage
"""

import time
from typing import TYPE_CHECKING

import psutil
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

if TYPE_CHECKING:
    from strata_fdtd.fdtd import FDTDSolver


def format_time(seconds: float) -> str:
    """Format time duration for display.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string like "1m 23s" or "2h 15m"
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs:02d}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes:02d}m"


def format_bytes(num_bytes: float) -> str:
    """Format byte count for display.

    Args:
        num_bytes: Number of bytes

    Returns:
        Formatted string like "1.5 GB" or "256 MB"
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


class SimulationProgress:
    """Real-time progress display for FDTD simulations.

    Shows a progress bar with simulation statistics including:
    - Current step and total steps
    - Elapsed time and ETA
    - Computational throughput (Mcells/s)
    - Current and peak memory usage

    Example:
        >>> progress = SimulationProgress(console, solver, num_steps)
        >>> solver.run(duration=1e-6, callback=progress.update)
    """

    def __init__(
        self, console: Console, solver: "FDTDSolver", num_steps: int, update_interval: float = 0.1
    ):
        """Initialize progress display.

        Args:
            console: Rich console instance
            solver: FDTD solver instance
            num_steps: Total number of timesteps
            update_interval: Minimum time between updates (seconds)
        """
        self.console = console
        self.solver = solver
        self.num_steps = num_steps
        self.update_interval = update_interval

        self.start_time = time.time()
        self.last_update = 0
        self.peak_memory = 0

        # Create progress bar
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
        )

        self.task = self.progress.add_task("Computing", total=num_steps)
        self.progress.start()

        # Stats display (updated below progress bar)
        self._stats_text = ""

    def update(self, step: int):
        """Update progress display for current timestep.

        This method is called by the solver after each timestep. It updates
        the progress bar and statistics, but rate-limits updates to avoid
        excessive overhead.

        Args:
            step: Current timestep number (0-indexed)
        """
        current_time = time.time()

        # Rate limit updates
        if current_time - self.last_update < self.update_interval:
            return

        # Update progress bar
        self.progress.update(self.task, completed=step + 1)

        # Compute statistics
        elapsed = current_time - self.start_time
        steps_completed = step + 1

        # Computational throughput
        if elapsed > 0 and steps_completed > 0:
            cells_per_second = (steps_completed * self.solver.grid.num_cells) / elapsed
            throughput_mcells = cells_per_second / 1e6
        else:
            throughput_mcells = 0

        # Memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        current_memory = memory_info.rss / (1024**3)  # GB
        self.peak_memory = max(self.peak_memory, current_memory)

        # Update stats display
        stats_parts = [
            f"Speed: {throughput_mcells:.1f} Mcells/s",
            f"Memory: {current_memory:.2f} GB",
            f"(peak: {self.peak_memory:.2f} GB)",
        ]

        # Clear previous stats line and print new one
        self.console.print("\r" + " | ".join(stats_parts), end="", style="dim")

        self.last_update = current_time

    def finish(self):
        """Finalize progress display.

        Call this after simulation completes to clean up the progress bar
        and show final statistics.
        """
        self.progress.stop()

        # Print final newline to clear stats line
        self.console.print()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


def print_simulation_info(console: Console, solver: "FDTDSolver", output_path, num_steps: int):
    """Print simulation parameters before running.

    Args:
        console: Rich console instance
        solver: FDTD solver instance
        output_path: Path to output file
        num_steps: Number of timesteps to run
    """
    grid = solver.grid
    num_cells = grid.num_cells

    # Create info table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")

    # Grid info
    shape_str = f"{grid.shape[0]} × {grid.shape[1]} × {grid.shape[2]}"
    cells_str = f"({num_cells/1e6:.1f}M cells)"
    table.add_row("Grid", f"{shape_str} {cells_str}")

    # Resolution
    if hasattr(grid, "is_uniform") and grid.is_uniform:
        res_str = f"{grid.resolution*1e3:.2f} mm"
    else:
        res_str = f"{grid.min_spacing*1e3:.2f}–{grid.max_spacing*1e3:.2f} mm"
    table.add_row("Resolution", res_str)

    # Timestep and duration
    table.add_row("Timestep", f"{solver.dt:.2e} s")
    total_time = solver.dt * num_steps
    table.add_row("Duration", f"{num_steps} steps ({total_time:.2e} s)")

    # Backend info
    backend = "native" if solver.using_native else "python"
    if hasattr(solver, "num_threads"):
        backend_str = f"{backend} ({solver.num_threads} threads)"
    else:
        backend_str = backend
    table.add_row("Backend", backend_str)

    # Output file
    table.add_row("Output", str(output_path))

    console.print(table)
    console.print()
