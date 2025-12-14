#!/usr/bin/env python3
"""
Migration script for ml-audio-codecs → strata-fdtd.

This script automates:
1. Copying files to new locations
2. Updating import statements
3. Renaming module references
"""
import shutil
import re
from pathlib import Path

# Source and destination roots
SRC_ROOT = Path("/Users/rwalters/GitHub/ml-audio-codecs")
DST_ROOT = Path("/Users/rwalters/GitHub/strata-fdtd")

# File mapping: (source_path, dest_path)
FILE_MAPPING = {
    # Core
    "src/metamaterial/fdtd.py": "src/strata_fdtd/core/solver.py",
    "src/metamaterial/fdtd_gpu.py": "src/strata_fdtd/core/solver_gpu.py",
    "src/metamaterial/grid.py": "src/strata_fdtd/core/grid.py",

    # Boundaries (will be split later)
    "src/metamaterial/boundaries.py": "src/strata_fdtd/boundaries.py",

    # Materials (directory copy)
    "src/metamaterial/materials/": "src/strata_fdtd/materials/",

    # Geometry
    "src/metamaterial/sdf.py": "src/strata_fdtd/geometry/sdf.py",
    "src/metamaterial/primitives.py": "src/strata_fdtd/geometry/primitives.py",
    "src/metamaterial/paths.py": "src/strata_fdtd/geometry/paths.py",
    "src/metamaterial/loudspeaker.py": "src/strata_fdtd/geometry/loudspeaker.py",
    "src/metamaterial/resonator.py": "src/strata_fdtd/geometry/resonator.py",
    "src/metamaterial/parametric.py": "src/strata_fdtd/geometry/parametric.py",
    "src/metamaterial/material_geometry.py": "src/strata_fdtd/geometry/material_assignment.py",

    # Manufacturing
    "src/metamaterial/geometry.py": "src/strata_fdtd/manufacturing/lamination.py",
    "src/metamaterial/constraints.py": "src/strata_fdtd/manufacturing/constraints.py",
    "src/metamaterial/sdf_conversion.py": "src/strata_fdtd/manufacturing/conversion.py",
    "src/metamaterial/export.py": "src/strata_fdtd/manufacturing/export.py",

    # I/O
    "src/metamaterial/hdf5_output.py": "src/strata_fdtd/io/hdf5.py",
    "src/metamaterial/fdtd_output.py": "src/strata_fdtd/io/output.py",
    "src/metamaterial/io.py": "src/strata_fdtd/io/legacy.py",

    # Analysis
    "src/metamaterial/weighting.py": "src/strata_fdtd/analysis/weighting.py",

    # CLI
    "src/metamaterial/cli/": "src/strata_fdtd/cli/",

    # Native kernels
    "src/metamaterial/_fdtd_kernels/": "src/strata_fdtd/_kernels/",

    # Tests (all files)
    "tests/metamaterial/": "tests/",

    # Examples
    "examples/": "examples/",

    # Docs (excluding codec specs)
    "docs/getting-started.md": "docs/getting-started.md",
    "docs/api-reference.md": "docs/api-reference.md",
    "docs/cli-reference.md": "docs/cli-reference.md",
    "docs/builder-guide.md": "docs/builder-guide.md",
    "docs/viewer-guide.md": "docs/viewer-guide.md",
    "docs/troubleshooting.md": "docs/troubleshooting.md",
    "docs/ci-cd.md": "docs/ci-cd.md",

    # Viz
    "viz/": "viz/",

    # Benchmarks
    "scripts/profile_fdtd.py": "benchmarks/profile_fdtd.py",
    "scripts/benchmark_ade.py": "benchmarks/benchmark_ade.py",
    "scripts/benchmark_gpu_pml.py": "benchmarks/benchmark_gpu_pml.py",
    "scripts/benchmark_hdf5.py": "benchmarks/benchmark_hdf5.py",
}

def update_imports(content: str) -> str:
    """Update import statements."""
    # Package name replacement
    content = re.sub(r'\bmetamaterial\b', 'strata_fdtd', content)

    # Module-specific renames
    content = re.sub(
        r'from strata_fdtd\.geometry import (Slice|Stack|Violation)',
        r'from strata_fdtd.manufacturing.lamination import \1',
        content
    )
    content = re.sub(
        r'from strata_fdtd import (export_dxf|export_stl)',
        r'from strata_fdtd.manufacturing.export import \1',
        content
    )
    content = re.sub(
        r'from strata_fdtd\.sdf_conversion',
        r'from strata_fdtd.manufacturing.conversion',
        content
    )
    content = re.sub(
        r'from strata_fdtd\.constraints',
        r'from strata_fdtd.manufacturing.constraints',
        content
    )
    content = re.sub(
        r'from strata_fdtd\.material_geometry',
        r'from strata_fdtd.geometry.material_assignment',
        content
    )
    content = re.sub(
        r'from strata_fdtd\.hdf5_output',
        r'from strata_fdtd.io.hdf5',
        content
    )
    content = re.sub(
        r'from strata_fdtd\.fdtd_output',
        r'from strata_fdtd.io.output',
        content
    )

    # C++ extension
    content = re.sub(r'_fdtd_kernels', '_kernels', content)

    return content

def copy_and_update_file(src: Path, dst: Path):
    """Copy file and update contents."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.suffix in ['.py', '.md', '.txt', '.yml', '.yaml', '.json', '.toml']:
        # Read, update, write
        try:
            content = src.read_text(encoding='utf-8')
            content = update_imports(content)
            dst.write_text(content, encoding='utf-8')
            print(f"✓ {src.relative_to(SRC_ROOT)} → {dst.relative_to(DST_ROOT)}")
        except Exception as e:
            print(f"✗ Error processing {src}: {e}")
    else:
        # Binary copy
        try:
            shutil.copy2(src, dst)
            print(f"✓ {src.relative_to(SRC_ROOT)} → {dst.relative_to(DST_ROOT)} (binary)")
        except Exception as e:
            print(f"✗ Error copying {src}: {e}")

def migrate():
    """Run migration."""
    print("Starting migration...\n")

    for src_path, dst_path in FILE_MAPPING.items():
        src = SRC_ROOT / src_path
        dst = DST_ROOT / dst_path

        if not src.exists():
            print(f"⚠ Source not found: {src_path}")
            continue

        if src.is_dir():
            # Copy directory recursively
            print(f"📁 Copying directory: {src_path}")

            for item in src.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(src)
                    dst_file = dst / rel_path
                    copy_and_update_file(item, dst_file)
        else:
            copy_and_update_file(src, dst)

    print("\n✓ Migration complete!")
    print("\nNext steps:")
    print("1. Create __init__.py files for subpackages")
    print("2. Split boundaries.py and sdf.py")
    print("3. Update pyproject.toml")
    print("4. Update CMakeLists.txt")
    print("5. Run tests")

if __name__ == "__main__":
    migrate()
