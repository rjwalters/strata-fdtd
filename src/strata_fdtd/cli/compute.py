"""Command-line tool for executing FDTD simulation scripts.

The fdtd-compute CLI tool executes simulation scripts with progress tracking
and HDF5 output generation. It provides a user-friendly interface for running
computationally intensive simulations.
"""

import hashlib
import sys
import time
from pathlib import Path

import click
from rich.console import Console

from .executor import RestrictedImportError, execute_simulation_script, validate_solver_object
from .progress import SimulationProgress, format_time, print_simulation_info

console = Console()


@click.command()
@click.argument("script", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file path (default: results_{hash}.h5)",
)
@click.option("--threads", "-t", type=int, help="Number of threads for native backend")
@click.option(
    "--backend",
    type=click.Choice(["auto", "native", "python"]),
    default="auto",
    help="Force specific backend (default: auto-detect)",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output with debug info")
@click.option("--dry-run", is_flag=True, help="Validate script without running simulation")
@click.option(
    "--snapshot-interval",
    type=int,
    help="Save pressure field every N steps to HDF5 (warning: increases file size)",
)
@click.version_option(version="0.1.0", prog_name="fdtd-compute")
def main(
    script: Path,
    output: Path | None,
    threads: int | None,
    backend: str,
    verbose: bool,
    dry_run: bool,
    snapshot_interval: int | None,
):
    """Execute FDTD simulation from Python script.

    SCRIPT is the path to a Python file that defines a 'solver' variable
    containing an FDTDSolver instance. The solver will be executed and
    results saved to an HDF5 file.

    Example script:

    \b
        from strata_fdtd import FDTDSolver, GaussianPulse
        solver = FDTDSolver(shape=(100, 100, 100), resolution=1e-3)
        solver.add_source(GaussianPulse(position=(0.05, 0.05, 0.05), frequency=40e3))
        solver.add_probe("center", (0.05, 0.05, 0.05))
        # solver.run() will be called by fdtd-compute

    The tool will:
    - Execute the script to create the solver
    - Display simulation parameters
    - Run the simulation with progress tracking
    - Save results to HDF5 with the source script embedded
    """
    try:
        # Read and hash script
        console.print(f"\n[bold]FDTD Simulation:[/bold] {script.name}", style="blue")
        console.print("─" * 60)

        script_content = script.read_text()
        script_hash = hashlib.sha256(script_content.encode()).hexdigest()

        if verbose:
            console.print(f"Script hash: {script_hash}")

        # Determine output path
        if output is None:
            output = Path(f"results_{script_hash[:8]}.h5")

        # Execute script to get solver
        console.print("Loading simulation...", style="dim")
        try:
            namespace = execute_simulation_script(script, script_content, verbose=verbose)
        except RestrictedImportError as e:
            console.print(f"\n[bold red]Security Error:[/bold red] {e}")
            console.print(
                "\n[yellow]Simulation scripts can only import:[/yellow] "
                "strata_fdtd, numpy, scipy, math, pathlib"
            )
            return 1
        except SyntaxError as e:
            console.print("\n[bold red]Syntax Error in script:[/bold red]")
            console.print(f"  {e}")
            return 1

        # Validate and extract solver
        try:
            solver = validate_solver_object(namespace)
        except ValueError as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
            return 1

        # Configure solver backend and threads
        if backend != "auto":
            if verbose:
                console.print(f"Forcing backend: {backend}")
            # Note: FDTDSolver doesn't have a direct backend setter, but we can check
            if backend == "native" and not solver.using_native:
                console.print(
                    "[yellow]Warning:[/yellow] Native backend not available, "
                    "using Python fallback"
                )

        if threads is not None:
            if hasattr(solver, "set_num_threads"):
                solver.set_num_threads(threads)
                if verbose:
                    console.print(f"Thread count set to: {threads}")
            else:
                console.print(
                    "[yellow]Warning:[/yellow] Solver doesn't support thread configuration"
                )

        # Determine number of steps
        # We need to look for 'duration' or 'num_steps' in the namespace
        # or use a default
        duration = namespace.get("duration")
        if duration is None:
            # Default to 1000 steps if not specified
            num_steps = namespace.get("num_steps", 1000)
            duration = num_steps * solver.dt
        else:
            num_steps = int(duration / solver.dt)

        # Show simulation info
        print_simulation_info(console, solver, output, num_steps)

        if dry_run:
            console.print("[yellow]Dry run - simulation not executed[/yellow]")
            return 0

        # Run simulation with progress tracking
        start_time = time.time()

        progress = SimulationProgress(console, solver, num_steps)

        try:
            solver.run(
                duration=duration,
                output_file=str(output),
                script_content=script_content,
                callback=progress.update,
                snapshot_interval=snapshot_interval,
            )
        except KeyboardInterrupt:
            progress.finish()
            console.print("\n[yellow]Interrupted by user[/yellow]")
            return 130  # Standard exit code for SIGINT
        except Exception as e:
            progress.finish()
            console.print(f"\n[bold red]Simulation Error:[/bold red] {e}")
            if verbose:
                console.print_exception()
            return 1
        finally:
            progress.finish()

        runtime = time.time() - start_time

        # Success summary
        console.print("─" * 60)
        console.print("✓ [bold green]Simulation complete![/bold green]")

        # Get file size
        if output.exists():
            file_size = output.stat().st_size
            console.print(f"  Output: {output} ({file_size / 1e6:.1f} MB)")
        else:
            console.print(f"  Output: {output}")

        console.print(f"  Runtime: {format_time(runtime)}")

        # Computational throughput
        total_cells = num_steps * solver.grid.num_cells
        throughput = total_cells / runtime / 1e6
        console.print(f"  Average throughput: {throughput:.1f} Mcells/s")

        if verbose:
            console.print("\n[dim]Results can be analyzed with HDF5 tools (h5py, HDFView)[/dim]")

        return 0

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        if verbose:
            console.print_exception()
        return 1


if __name__ == "__main__":
    sys.exit(main())
