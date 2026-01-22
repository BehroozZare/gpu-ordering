#!/bin/bash
#
# Patch Visualization - Shell wrapper
#
# Renders a mesh with vertices colored by their patch assignments.
# Each patch gets a distinct pastel color for clear visualization.
#
# Usage:
#   ./run_render_patch.sh                                    # Render default mesh (beetle)
#   ./run_render_patch.sh --mesh /path/to/mesh.obj           # Render specific mesh
#   ./run_render_patch.sh --patch_data /path/to/patches.txt  # Use specific patch data
#   ./run_render_patch.sh --help                             # Show all options
#
# Options:
#   --mesh PATH         Path to mesh file (OBJ, PLY)
#   --patch_data PATH   Path to vertex-to-patch mapping file
#   --output PATH       Output image path
#   --subdivision N     Subdivision level (0-2)
#   --resolution N      Output resolution (square)
#   --samples N         Render samples

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_SCRIPT="$SCRIPT_DIR/render_patch.py"

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

# Check if render script exists
if [ ! -f "$PATCH_SCRIPT" ]; then
    echo "Error: render_patch.py not found at $PATCH_SCRIPT"
    exit 1
fi

echo "Running patch visualization..."
echo "Script: $PATCH_SCRIPT"
echo "Python: $PYTHON"
echo ""

# Run the render script directly with Python (bpy is installed as a package)
$PYTHON "$PATCH_SCRIPT" "$@"
