"""
Export functionality for strata_fdtd geometry.

Provides DXF (laser cutting), JSON (Three.js visualization), and STL (3D printing)
export formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from strata_fdtd.manufacturing.lamination import Stack


def export_dxf(
    stack: Stack,
    output_dir: Path,
    kerf_compensation: float = 0.0,
) -> list[Path]:
    """
    Export each slice as DXF for laser cutting.

    Creates one DXF file per slice with:
    - Closed polylines for all cut paths
    - Layer naming: "cut" for cutting paths
    - Optional kerf compensation (offset cut paths outward)

    Args:
        stack: The Stack to export
        output_dir: Directory to write DXF files
        kerf_compensation: Amount to offset cut paths outward (half kerf width)

    Returns:
        List of paths to created DXF files
    """
    try:
        import ezdxf
    except ImportError as err:
        raise ImportError(
            "ezdxf is required for DXF export. "
            "Install with: pip install ezdxf"
        ) from err

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created_files = []

    for slice_ in stack.slices:
        # Create new DXF document
        doc = ezdxf.new("R2010")  # AutoCAD 2010 format for compatibility
        msp = doc.modelspace()

        # Add layers
        doc.layers.add("cut", color=7)  # White for cutting

        # Get boundary polygons
        polygons = slice_.boundary_polygons()

        for polygon in polygons:
            if len(polygon) < 3:
                continue

            # Apply kerf compensation if specified
            if kerf_compensation > 0:
                polygon = _offset_polygon(polygon, kerf_compensation)

            # Close the polygon
            if not np.allclose(polygon[0], polygon[-1]):
                polygon = np.vstack([polygon, polygon[0]])

            # Convert to list of tuples for ezdxf
            points = [(float(p[0]), float(p[1])) for p in polygon]

            # Add as lightweight polyline
            msp.add_lwpolyline(
                points,
                dxfattribs={"layer": "cut"},
                close=True,
            )

        # Save file
        filename = f"slice_{slice_.z_index:03d}.dxf"
        filepath = output_dir / filename
        doc.saveas(filepath)
        created_files.append(filepath)

    return created_files


def _offset_polygon(polygon: np.ndarray, offset: float) -> np.ndarray:
    """
    Offset a polygon outward by a given amount.

    Uses a simple approach suitable for laser cutting:
    - Compute edge normals
    - Move vertices along average of adjacent normals

    Args:
        polygon: Nx2 array of polygon vertices
        offset: Distance to offset (positive = outward)

    Returns:
        Offset polygon as Nx2 array
    """
    n = len(polygon)
    if n < 3:
        return polygon

    # Ensure polygon is closed for calculation
    closed = np.vstack([polygon, polygon[0], polygon[1]])

    # Compute edge normals (perpendicular to edges, pointing outward)
    normals = []
    for i in range(n):
        edge = closed[i + 1] - closed[i]
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-10:
            normals.append(np.array([0.0, 0.0]))
        else:
            # Normal perpendicular to edge
            normal = np.array([-edge[1], edge[0]]) / edge_len
            normals.append(normal)

    normals = np.array(normals)

    # Determine winding direction (CCW = outer boundary = offset outward)
    # Use signed area
    signed_area = 0.0
    for i in range(n):
        j = (i + 1) % n
        signed_area += polygon[i, 0] * polygon[j, 1]
        signed_area -= polygon[j, 0] * polygon[i, 1]

    # If clockwise (negative area), flip offset direction
    if signed_area < 0:
        offset = -offset

    # Offset each vertex along average of adjacent normals
    offset_polygon = np.zeros_like(polygon)
    for i in range(n):
        prev_normal = normals[(i - 1) % n]
        curr_normal = normals[i]

        # Average normal (bisector direction)
        avg_normal = prev_normal + curr_normal
        avg_len = np.linalg.norm(avg_normal)

        if avg_len < 1e-10:
            # Parallel edges, use one normal
            avg_normal = curr_normal
        else:
            avg_normal = avg_normal / avg_len

            # Adjust offset distance for corner angle
            # (offset further at sharp corners to maintain edge distance)
            dot = np.dot(prev_normal, curr_normal)
            if dot > -0.999:  # Not a hairpin turn
                # Miter factor
                miter = 1.0 / np.sqrt((1 + dot) / 2)
                avg_normal *= min(miter, 2.0)  # Limit to avoid extreme miter

        offset_polygon[i] = polygon[i] + offset * avg_normal

    return offset_polygon


def export_json(stack: Stack, path: Path) -> None:
    """
    Export stack as JSON for Three.js visualization.

    Creates a JSON file with:
    - Resolution and thickness information
    - Per-slice polygon data
    - Metadata for rendering

    Args:
        stack: The Stack to export
        path: Path to output JSON file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bbox = stack.bounding_box

    data = {
        "resolution_xy": stack.resolution_xy,
        "thickness_z": stack.thickness_z,
        "slices": [],
        "metadata": {
            "total_height": stack.height_mm,
            "bounding_box": list(bbox),
            "num_slices": stack.num_slices,
        },
    }

    for slice_ in stack.slices:
        polygons = slice_.boundary_polygons()

        # Convert polygons to serializable format
        polygon_data = []
        for polygon in polygons:
            # Ensure proper winding order
            # CCW for outer boundaries, CW for holes
            # (Three.js expects CCW for front-facing)
            polygon_list = polygon.tolist()

            # Close polygon if not already closed
            if polygon_list and polygon_list[0] != polygon_list[-1]:
                polygon_list.append(polygon_list[0])

            polygon_data.append(polygon_list)

        slice_data = {
            "z_index": slice_.z_index,
            "z_mm": slice_.z_index * stack.thickness_z,
            "polygons": polygon_data,
        }
        data["slices"].append(slice_data)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def export_stl(stack: Stack, path: Path) -> None:
    """
    Export stack as STL mesh for 3D visualization/printing.

    Creates a triangulated mesh from the slice stack with:
    - Proper outward-facing normals
    - Watertight mesh (closed surfaces)

    Args:
        stack: The Stack to export
        path: Path to output STL file
    """
    try:
        from stl import mesh as stl_mesh
    except ImportError as err:
        raise ImportError(
            "numpy-stl is required for STL export. "
            "Install with: pip install numpy-stl"
        ) from err

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all triangles
    vertices = []
    faces = []

    for i, slice_ in enumerate(stack.slices):
        z_bottom = i * stack.thickness_z
        z_top = (i + 1) * stack.thickness_z

        polygons = slice_.boundary_polygons()

        for polygon in polygons:
            if len(polygon) < 3:
                continue

            # Ensure polygon is closed
            if not np.allclose(polygon[0], polygon[-1]):
                polygon = np.vstack([polygon, polygon[0]])

            n_pts = len(polygon) - 1  # Exclude duplicate closing point

            # Add vertices for top and bottom of this polygon
            base_idx = len(vertices)

            for j in range(n_pts):
                # Bottom vertex
                vertices.append([polygon[j, 0], polygon[j, 1], z_bottom])
                # Top vertex
                vertices.append([polygon[j, 0], polygon[j, 1], z_top])

            # Create faces for the extruded walls
            for j in range(n_pts):
                next_j = (j + 1) % n_pts

                # Indices in vertices array
                bl = base_idx + j * 2  # bottom left
                tl = base_idx + j * 2 + 1  # top left
                br = base_idx + next_j * 2  # bottom right
                tr = base_idx + next_j * 2 + 1  # top right

                # Two triangles per quad (CCW winding for outward normal)
                faces.append([bl, tl, tr])
                faces.append([bl, tr, br])

            # Cap faces (top and bottom)
            # Use fan triangulation from first vertex
            if n_pts >= 3:
                # Bottom cap (reverse winding for downward normal)
                for j in range(1, n_pts - 1):
                    v0 = base_idx + 0
                    v1 = base_idx + (j + 1) * 2
                    v2 = base_idx + j * 2
                    faces.append([v0, v1, v2])

                # Top cap (normal winding for upward normal)
                for j in range(1, n_pts - 1):
                    v0 = base_idx + 1
                    v1 = base_idx + j * 2 + 1
                    v2 = base_idx + (j + 1) * 2 + 1
                    faces.append([v0, v1, v2])

    if not vertices or not faces:
        # Create empty/minimal mesh
        empty_mesh = stl_mesh.Mesh(np.zeros(0, dtype=stl_mesh.Mesh.dtype))
        empty_mesh.save(str(path))
        return

    # Convert to numpy arrays
    vertices = np.array(vertices)
    faces = np.array(faces)

    # Create the mesh
    mesh_data = stl_mesh.Mesh(np.zeros(len(faces), dtype=stl_mesh.Mesh.dtype))

    for i, face in enumerate(faces):
        for j in range(3):
            mesh_data.vectors[i][j] = vertices[face[j]]

    # Save to file
    mesh_data.save(str(path))


def export_slice_svg(
    slice_,
    path: Path,
    stroke_width: float = 0.5,
    stroke_color: str = "#000000",
    fill_color: str = "none",
) -> None:
    """
    Export a single slice as SVG for preview/documentation.

    Args:
        slice_: The Slice to export
        path: Path to output SVG file
        stroke_width: Line width in mm
        stroke_color: CSS color for stroke
        fill_color: CSS color for fill (or "none")
    """
    from strata_fdtd.manufacturing.lamination import Slice

    if not isinstance(slice_, Slice):
        raise TypeError("Expected Slice object")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    w_mm, h_mm = slice_.shape_mm
    polygons = slice_.boundary_polygons()

    # Build SVG
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w_mm}mm" height="{h_mm}mm" '
        f'viewBox="0 0 {w_mm} {h_mm}">',
    ]

    for polygon in polygons:
        if len(polygon) < 2:
            continue

        # Build path data
        path_data = f"M {polygon[0, 0]:.3f} {polygon[0, 1]:.3f}"
        for pt in polygon[1:]:
            path_data += f" L {pt[0]:.3f} {pt[1]:.3f}"
        path_data += " Z"

        svg_parts.append(
            f'  <path d="{path_data}" '
            f'stroke="{stroke_color}" '
            f'stroke-width="{stroke_width}" '
            f'fill="{fill_color}"/>'
        )

    svg_parts.append("</svg>")

    with open(path, "w") as f:
        f.write("\n".join(svg_parts))
