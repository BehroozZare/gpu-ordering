#!/bin/bash
#
# Patch Graph Visualization - Shell wrapper
#
# Renders a graph where each patch is a node (sphere) and edges connect
# adjacent patches. Nodes are positioned at patch centroids in 3D space.
#
# Usage:
#   ./run_render_graph.sh                                    # Render default mesh (beetle)
#   ./run_render_graph.sh --mesh /path/to/mesh.obj           # Render specific mesh
#   ./run_render_graph.sh --patch_data /path/to/patches.txt  # Use specific patch data
#   ./run_render_graph.sh --help                             # Show all options
#
# Options:
#   --mesh PATH         Path to mesh file (OBJ, PLY)
#   --patch_data PATH   Path to vertex-to-patch mapping file
#   --output PATH       Output image path
#   --resolution N      Output resolution (square)
#   --samples N         Render samples
#   --node_scale F      Scale factor for node sphere radius (default: 1.0)
#   --edge_scale F      Scale factor for edge cylinder radius (default: 1.0)
#   --show_mesh         Show original mesh as wireframe overlay

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPH_SCRIPT="$SCRIPT_DIR/render_graph_from_patch.py"

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
if [ ! -f "$GRAPH_SCRIPT" ]; then
    echo "Error: render_graph_from_patch.py not found at $GRAPH_SCRIPT"
    exit 1
fi

echo "Running patch graph visualization..."
echo "Script: $GRAPH_SCRIPT"
echo "Python: $PYTHON"
echo ""

# Run the render script directly with Python (bpy is installed as a package)
$PYTHON "$GRAPH_SCRIPT" "$@"
