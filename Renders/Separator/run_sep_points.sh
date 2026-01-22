#!/bin/bash
#
# Separator Mesh Visualization (Spheres) - Shell wrapper
#
# Usage:
#   ./run_sep_points.sh                              # Render default mesh (squirrel)
#   ./run_sep_points.sh --mesh /path/to/mesh.obj     # Render specific mesh
#   ./run_sep_points.sh --max_level 1                # Show more separator levels
#   ./run_sep_points.sh --help                       # Show all options
#
# Options:
#   --mesh PATH             Path to mesh file (OBJ, PLY)
#   --etree PATH            Path to PARTH_etree_nodes file
#   --assigned PATH         Path to PARTH_assigned_nodes file
#   --output PATH           Output image path
#   --max_level N           Separator level (0=root, 1=root+level1, etc.)
#   --separator_radius N    Radius of separator spheres
#   --subdivision N         Subdivision level (0-2)
#   --resolution N          Output resolution (square)
#   --samples N             Render samples

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEP_SCRIPT="$SCRIPT_DIR/show_separator_as_point.py"

# Use the blender_env virtual environment with bpy installed
BLENDER_ENV="/home/behrooz/Desktop/Last_Project/blender_env"
PYTHON="$BLENDER_ENV/bin/python"

# Check if the virtual environment exists
if [ ! -f "$PYTHON" ]; then
    echo "Error: blender_env not found at $BLENDER_ENV"
    echo "Please create it with: python3.11 -m venv $BLENDER_ENV"
    echo "Then install: pip install bpy==4.3.0 --extra-index-url https://download.blender.org/pypi/"
    exit 1
fi

# Check if separator script exists
if [ ! -f "$SEP_SCRIPT" ]; then
    echo "Error: show_separator_as_point.py not found at $SEP_SCRIPT"
    exit 1
fi

echo "Running separator visualization (spheres)..."
echo "Script: $SEP_SCRIPT"
echo "Python: $PYTHON"
echo ""

# Run the separator script directly with Python (bpy is installed as a package)
$PYTHON "$SEP_SCRIPT" "$@"
