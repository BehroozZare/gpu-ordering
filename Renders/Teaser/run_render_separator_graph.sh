#!/bin/bash
#
# Separator Split Patch Graph Visualization - Shell wrapper
#
# Renders two separate patch graphs - one for the left side and one for the
# right side of the mesh separator. Patches spanning the separator appear in both.
#
# Usage:
#   ./run_render_separator_graph.sh                                    # Render default mesh (beetle)
#   ./run_render_separator_graph.sh --mesh /path/to/mesh.obj           # Render specific mesh
#   ./run_render_separator_graph.sh --patch_data /path/to/patches.txt  # Use specific patch data
#   ./run_render_separator_graph.sh --help                             # Show all options
#
# Options:
#   --mesh PATH           Path to mesh file (OBJ, PLY)
#   --patch_data PATH     Path to vertex-to-patch mapping file
#   --etree PATH          Path to PARTH_etree_nodes file
#   --assigned PATH       Path to PARTH_assigned_nodes file
#   --output_prefix PATH  Output prefix (creates _left.png and _right.png)
#   --resolution N        Output resolution (square)
#   --samples N           Render samples
#   --node_scale F        Scale factor for node sphere radius (default: 1.0)
#   --edge_scale F        Scale factor for edge cylinder radius (default: 1.0)
#
# Output:
#   {output_prefix}_left.png   - Graph of patches on the left side
#   {output_prefix}_right.png  - Graph of patches on the right side
#   {output_prefix}_left.blend - Blender file for left graph
#   {output_prefix}_right.blend - Blender file for right graph

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEP_GRAPH_SCRIPT="$SCRIPT_DIR/render_separator_in_patch_graph.py"

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
if [ ! -f "$SEP_GRAPH_SCRIPT" ]; then
    echo "Error: render_separator_in_patch_graph.py not found at $SEP_GRAPH_SCRIPT"
    exit 1
fi

echo "Running separator split patch graph visualization..."
echo "Script: $SEP_GRAPH_SCRIPT"
echo "Python: $PYTHON"
echo ""

# Run the render script directly with Python (bpy is installed as a package)
$PYTHON "$SEP_GRAPH_SCRIPT" "$@"
