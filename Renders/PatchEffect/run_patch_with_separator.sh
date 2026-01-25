#!/bin/bash
#
# Patch with Separator Visualization - Shell wrapper
#
# Renders mesh with patch-colored vertices and separator spheres.
#
# Usage:
#   ./run_patch_with_separator.sh --mesh /path/to/mesh.obj --num_patches 8
#   ./run_patch_with_separator.sh --mesh /path/to/mesh.obj --num_patches 16 --max_level 1
#   ./run_patch_with_separator.sh --help
#
# Options:
#   --mesh PATH             Path to mesh file (OBJ, PLY)
#   --patch_data PATH       Path to vertex-to-patch mapping file
#   --etree PATH            Path to PATCH_ORDERING_etree_nodes file
#   --assigned PATH         Path to PATCH_ORDERING_assigned_nodes file
#   --output PATH           Output image path
#   --num_patches N         Number of patches (4, 8, 16, etc.) - for auto file paths
#   --max_level N           Separator level (0=root, 1=root+level1, etc.)
#   --subdivision N         Subdivision level (0-2)
#   --resolution N          Output resolution (square)
#   --samples N             Render samples

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/patch_with_separator.py"

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

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: patch_with_separator.py not found at $PYTHON_SCRIPT"
    exit 1
fi

echo "Running patch with separator visualization..."
echo "Script: $PYTHON_SCRIPT"
echo "Python: $PYTHON"
echo ""

# Run the Python script with bpy
$PYTHON "$PYTHON_SCRIPT" "$@"
