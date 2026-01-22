#!/usr/bin/env python3
"""
Patch Visualization using BlenderToolbox

Renders a mesh with vertices colored by their patch assignments.
Each patch gets a distinct high-contrast color for clear visualization.

Requires:
    - pip install bpy==4.3.0 --extra-index-url https://download.blender.org/pypi/
    - pip install blendertoolbox

Usage:
    python render_patch.py --mesh /path/to/mesh.obj --patch_data /path/to/patches.txt
    python render_patch.py --help
"""

import bpy
import os
import sys
import argparse
import blendertoolbox as bt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Render mesh with patch-colored vertices')
    parser.add_argument('--mesh', type=str,
                        default='/media/behrooz/FarazHard/Last_Project/BenchmarkMesh/tri-mesh/final/beetle.obj',
                        help='Path to the mesh file (OBJ, PLY, etc.)')
    parser.add_argument('--patch_data', type=str, default=None,
                        help='Path to vertex-to-patch mapping file (one patch ID per line)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path (auto-generated if not specified)')
    parser.add_argument('--resolution', type=int, default=1080,
                        help='Output resolution (square image)')
    parser.add_argument('--samples', type=int, default=200,
                        help='Number of render samples (higher=better quality)')
    parser.add_argument('--subdivision', type=int, default=1,
                        help='Subdivision level for smoother surface (0=none, 1=light, 2=smooth)')
    return parser.parse_args()


# High-contrast colors for patches (11 distinct colors for patches 0-10)
PATCH_COLORS = [
    (0.122, 0.467, 0.706, 1.0),  # Strong blue
    (1.000, 0.498, 0.055, 1.0),  # Bright orange
    (0.173, 0.627, 0.173, 1.0),  # Forest green
    (0.839, 0.153, 0.157, 1.0),  # Crimson red
    (0.580, 0.404, 0.741, 1.0),  # Rich purple
    (0.549, 0.337, 0.294, 1.0),  # Brown
    (0.890, 0.467, 0.761, 1.0),  # Hot pink
    (0.498, 0.498, 0.498, 1.0),  # Medium gray
    (0.737, 0.741, 0.133, 1.0),  # Olive yellow
    (0.090, 0.745, 0.812, 1.0),  # Cyan
    (0.682, 0.780, 0.910, 1.0),  # Steel blue
]


def load_patch_ids(filepath):
    """Load patch IDs from file (one integer per line)."""
    with open(filepath, 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]


def apply_patch_colors(mesh_obj, patch_ids):
    """
    Apply vertex colors based on patch assignments.
    
    Args:
        mesh_obj: Blender mesh object
        patch_ids: List of patch IDs (index i = patch ID for vertex i)
    
    Returns:
        Dictionary with patch statistics
    """
    mesh_data = mesh_obj.data
    
    # Remove existing color attribute if present
    if "PatchColors" in mesh_data.color_attributes:
        mesh_data.color_attributes.remove(mesh_data.color_attributes["PatchColors"])
    
    # Create new color attribute with CORNER domain (per loop vertex)
    color_layer = mesh_data.color_attributes.new(
        name="PatchColors",
        type='FLOAT_COLOR',
        domain='CORNER'
    )
    
    # Count patches
    patch_counts = {}
    num_vertices = len(mesh_data.vertices)
    
    # Verify patch_ids length matches vertices
    if len(patch_ids) != num_vertices:
        print(f"Warning: patch_ids length ({len(patch_ids)}) != mesh vertices ({num_vertices})")
        print(f"Using min of both: {min(len(patch_ids), num_vertices)}")
    
    # Build vertex color mapping
    vertex_colors = {}
    for i in range(min(len(patch_ids), num_vertices)):
        patch_id = patch_ids[i]
        color = PATCH_COLORS[patch_id % len(PATCH_COLORS)]
        vertex_colors[i] = color
        patch_counts[patch_id] = patch_counts.get(patch_id, 0) + 1
    
    # Assign colors to each loop (face corner)
    for poly in mesh_data.polygons:
        for loop_idx in poly.loop_indices:
            vert_idx = mesh_data.loops[loop_idx].vertex_index
            color = vertex_colors.get(vert_idx, PATCH_COLORS[0])
            color_layer.data[loop_idx].color = color
    
    return patch_counts


def create_patch_color_material(mesh_obj):
    """
    Create an opaque Principled BSDF material that reads vertex colors.
    Uses visual settings from show_separator_as_point.py for better appearance.
    """
    mat = bpy.data.materials.new(name="PatchColorMaterial")
    mat.use_nodes = True
    
    # Ensure opaque render settings
    if hasattr(mat, "blend_method"):
        mat.blend_method = 'OPAQUE'
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = 'OPAQUE'
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Output node
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (400, 0)
    
    # Principled BSDF with good visual settings (from show_separator_as_point.py)
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (100, 0)
    
    # Set material properties for nice soft appearance
    if 'Roughness' in principled.inputs:
        principled.inputs['Roughness'].default_value = 0.45
    if 'Metallic' in principled.inputs:
        principled.inputs['Metallic'].default_value = 0.0
    if 'Alpha' in principled.inputs:
        principled.inputs['Alpha'].default_value = 1.0
    if 'Specular IOR Level' in principled.inputs:
        principled.inputs['Specular IOR Level'].default_value = 0.25
    
    # Attribute node to read vertex colors
    attr_node = nodes.new(type='ShaderNodeAttribute')
    attr_node.location = (-200, 0)
    attr_node.attribute_name = "PatchColors"
    attr_node.attribute_type = 'GEOMETRY'
    
    # Connect: VertexColor -> Base Color -> Output
    links.new(attr_node.outputs['Color'], principled.inputs['Base Color'])
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    # Assign material to mesh
    if mesh_obj.data.materials:
        mesh_obj.data.materials[0] = mat
    else:
        mesh_obj.data.materials.append(mat)
    
    return mat


def render_patches(mesh_path, patch_data_path, output_path, resolution=1080,
                   samples=200, subdivision=1):
    """
    Main rendering function for patch visualization.
    
    Args:
        mesh_path: Path to mesh file
        patch_data_path: Path to vertex-to-patch mapping file
        output_path: Output image path
        resolution: Output resolution (square)
        samples: Render samples
        subdivision: Subdivision level (0-2)
    """
    # Initialize Blender (same settings as render_mesh.py)
    bt.blenderInit(resolution, resolution, samples, 1.5)
    
    # Load mesh with same transforms as render_mesh.py
    location = (0, 0.09, 0.68)
    mesh_scale = (2, 2, 2)
    rotation = (90, 0, 50)
    mesh = bt.readMesh(mesh_path, location, rotation, mesh_scale)
    
    # Smooth shading
    bpy.ops.object.shade_smooth()
    
    # Load and apply patch colors
    print(f"Loading patch data from: {patch_data_path}")
    patch_ids = load_patch_ids(patch_data_path)
    print(f"Loaded {len(patch_ids)} patch assignments")
    
    patch_counts = apply_patch_colors(mesh, patch_ids)
    print(f"Applied colors for {len(patch_counts)} patches:")
    for pid in sorted(patch_counts.keys()):
        print(f"  Patch {pid}: {patch_counts[pid]} vertices")
    
    # Apply subdivision for smoother surface (before material)
    if subdivision > 0:
        bt.subdivision(mesh, level=subdivision)
    
    # Create and apply vertex color material
    create_patch_color_material(mesh)
    
    # Setup scene (same as render_mesh.py)
    bt.invisibleGround(shadowBrightness=0.9)
    
    # Set camera
    camLocation = (3, 0, 2)
    lookAtLocation = (0, 0, 0.5)
    focalLength = 45
    cam = bt.setCamera(camLocation, lookAtLocation, focalLength)
    
    # Set sun light
    lightAngle = (6, -30, -155)
    strength = 2
    shadowSoftness = 0.3
    bt.setLight_sun(lightAngle, strength, shadowSoftness)
    
    # Set ambient light
    bt.setLight_ambient(color=(0.1, 0.1, 0.1, 1))
    
    # Set shadow threshold for clean shadows
    bt.shadowThreshold(alphaThreshold=0.05, interpolationMode='CARDINAL')
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save .blend file alongside the image
    blend_path = os.path.splitext(output_path)[0] + '.blend'
    bpy.ops.wm.save_mainfile(filepath=blend_path)
    print(f"Saved Blender file: {blend_path}")
    
    # Render image
    bt.renderImage(output_path, cam)
    print(f"Rendered image: {output_path}")


def main():
    args = parse_args()
    
    # Validate mesh path
    if not os.path.exists(args.mesh):
        print(f"Error: Mesh file not found: {args.mesh}")
        sys.exit(1)
    
    # Generate default patch data path if not specified
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_name = os.path.splitext(os.path.basename(args.mesh))[0]
    
    if args.patch_data is None:
        # Default: look in output/render_data/
        args.patch_data = os.path.join(
            script_dir, '..', '..', 'output', 'render_data',
            f'{mesh_name}_vertex_to_patch.txt'
        )
    
    if not os.path.exists(args.patch_data):
        print(f"Error: Patch data file not found: {args.patch_data}")
        sys.exit(1)
    
    # Generate output path if not specified
    if args.output is None:
        figures_dir = os.path.join(script_dir, 'Figures')
        args.output = os.path.join(figures_dir, f'{mesh_name}_patches.png')
    
    print("=" * 60)
    print("Patch Visualization")
    print("=" * 60)
    print(f"Mesh: {args.mesh}")
    print(f"Patch Data: {args.patch_data}")
    print(f"Output: {args.output}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Samples: {args.samples}")
    print(f"Subdivision: {args.subdivision}")
    print("=" * 60)
    
    render_patches(
        mesh_path=args.mesh,
        patch_data_path=args.patch_data,
        output_path=args.output,
        resolution=args.resolution,
        samples=args.samples,
        subdivision=args.subdivision
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
