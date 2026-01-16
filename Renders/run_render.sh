#!/bin/bash
#
# Simple Mesh Renderer - Shell wrapper
#
# Usage:
#   ./run_render.sh                           # Render default mesh (squirrel)
#   ./run_render.sh --mesh /path/to/mesh.obj  # Render specific mesh
#   ./run_render.sh --color blue              # Use blue color preset
#   ./run_render.sh --help                    # Show all options
#
# Options:
#   --mesh PATH         Path to mesh file (OBJ, PLY)
#   --output PATH       Output image path
#   --color PRESET      Color: green, blue, orange, pink, gray
#   --subdivision N     Subdivision level (0-2)
#   --resolution N      Output resolution (square)
#   --samples N         Render samples
#   --rotation X Y Z    Mesh rotation in degrees
#   --scale N           Mesh scale factor

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RENDER_SCRIPT="$SCRIPT_DIR/render_mesh.py"

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
if [ ! -f "$RENDER_SCRIPT" ]; then
    echo "Error: render_mesh.py not found at $RENDER_SCRIPT"
    exit 1
fi

echo "Running mesh renderer..."
echo "Script: $RENDER_SCRIPT"
echo "Python: $PYTHON"
echo ""

# Run the render script directly with Python (bpy is installed as a package)
$PYTHON "$RENDER_SCRIPT" "$@"
