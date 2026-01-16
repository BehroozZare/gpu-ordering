#!/usr/bin/env python3
"""
Separator Mesh Visualization using Vertex Colors

Visualizes mesh partitioning based on elimination tree (etree) node labels.
Separators are shown as dark-colored vertices, non-separators as light grey.

Requires two data files:
    - PARTH_etree_nodes_{mesh}.txt: etree node IDs for each vertex
    - PARTH_assigned_nodes_{mesh}.txt: original vertex indices (mapping)

Usage:
    python show_separator_as_color.py --mesh /path/to/mesh.obj
    python show_separator_as_color.py --mesh /path/to/mesh.obj --etree /path/to/etree.txt --assigned /path/to/assigned.txt
"""

import bpy
import os
import sys
import argparse
import blendertoolbox as bt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize mesh separators with vertex colors')
    parser.add_argument('--mesh', type=str,
                        default='/media/behrooz/FarazHard/Last_Project/BenchmarkMesh/tri-mesh/final/squirrel.obj',
                        help='Path to the mesh file')
    parser.add_argument('--etree', type=str, default=None,
                        help='Path to PARTH_etree_nodes file')
    parser.add_argument('--assigned', type=str, default=None,
                        help='Path to PARTH_assigned_nodes file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path')
    parser.add_argument('--max_level', type=int, default=0,
                        help='Maximum separator level (0=root only, 1=root+level1, etc.)')
    parser.add_argument('--resolution', type=int, default=1080,
                        help='Output resolution')
    parser.add_argument('--samples', type=int, default=200,
                        help='Number of render samples')
    parser.add_argument('--subdivision', type=int, default=1,
                        help='Subdivision level for smoother surface (0=none, 1=light, 2=smooth)')
    return parser.parse_args()


# Color definitions for separator visualization
SEPARATOR_COLOR = (0.1, 0.1, 0.1, 1.0)        # Near-black for separators
NON_SEPARATOR_COLOR = (0.75, 0.75, 0.75, 1.0)  # Light grey for non-separators


def load_file_as_ints(filepath):
    """Load a file with one integer per line."""
    with open(filepath, 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]


def get_separator_vertices(etree_nodes, assigned_nodes, max_level):
    """
    Get separator vertex indices for the given level.
    
    Args:
        etree_nodes: List of etree node IDs for each position
        assigned_nodes: List of original vertex indices for each position
        max_level: Maximum separator level
    
    Returns:
        Set of original vertex indices that are separators
    
    Level 0: etree == 0
    Level 1: etree in {0, 1, 2}
    Level N: etree in range [0, 2^(N+1) - 2]
    """
    max_etree_id = (1 << (max_level + 1)) - 2  # 2^(max_level+1) - 2
    
    separator_verts = set()
    for i, (etree_id, orig_vertex) in enumerate(zip(etree_nodes, assigned_nodes)):
        if etree_id <= max_etree_id:
            separator_verts.add(orig_vertex)
    
    return separator_verts


def apply_vertex_colors(mesh_obj, etree_nodes, assigned_nodes, max_level):
    """
    Apply vertex colors to the mesh based on separator status.
    Separators get dark color, non-separators get light grey.
    """
    separator_verts = get_separator_vertices(etree_nodes, assigned_nodes, max_level)
    
    # Create a color attribute on the mesh (per-vertex colors)
    mesh_data = mesh_obj.data
    
    # Remove existing color attribute if present
    if "SeparatorColors" in mesh_data.color_attributes:
        mesh_data.color_attributes.remove(mesh_data.color_attributes["SeparatorColors"])
    
    # Create new color attribute with CORNER domain (per loop vertex)
    # This is the standard way to do vertex colors in Blender 4.x
    color_layer = mesh_data.color_attributes.new(
        name="SeparatorColors",
        type='FLOAT_COLOR',
        domain='CORNER'  # Per-loop (face corner) colors
    )
    
    # Build a mapping from vertex index to color
    vertex_colors = {}
    for i in range(len(mesh_data.vertices)):
        if i in separator_verts:
            vertex_colors[i] = SEPARATOR_COLOR
        else:
            vertex_colors[i] = NON_SEPARATOR_COLOR
    
    # Assign colors to each loop (face corner)
    for poly in mesh_data.polygons:
        for loop_idx in poly.loop_indices:
            vert_idx = mesh_data.loops[loop_idx].vertex_index
            color_layer.data[loop_idx].color = vertex_colors.get(vert_idx, NON_SEPARATOR_COLOR)
    
    print(f"Applied vertex colors: {len(separator_verts)} separators (dark), "
          f"{len(mesh_data.vertices) - len(separator_verts)} non-separators (light)")
    
    return len(separator_verts)


def create_vertex_color_material(mesh_obj):
    """
    Create a material that uses the vertex color attribute.
    """
    mat = bpy.data.materials.new(name="SeparatorColorMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Create nodes
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (400, 0)
    
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (100, 0)
    principled.inputs['Roughness'].default_value = 0.5
    principled.inputs['Specular IOR Level'].default_value = 0.3
    
    # Attribute node to read vertex colors
    attr_node = nodes.new(type='ShaderNodeAttribute')
    attr_node.location = (-200, 0)
    attr_node.attribute_name = "SeparatorColors"
    attr_node.attribute_type = 'GEOMETRY'
    
    # Connect nodes
    links.new(attr_node.outputs['Color'], principled.inputs['Base Color'])
    links.new(principled.outputs['BSDF'], output_node.inputs['Surface'])
    
    # Assign material to mesh
    if mesh_obj.data.materials:
        mesh_obj.data.materials[0] = mat
    else:
        mesh_obj.data.materials.append(mat)
    
    return mat


def render_separators_colored(mesh_path, etree_path, assigned_path, output_path, max_level=0,
                               resolution=1080, samples=200, subdivision=1):
    """Main rendering function with vertex-colored separators."""
    
    # Initialize Blender
    bt.blenderInit(resolution, resolution, samples, 1.5)
    
    # Load mesh with same transforms as other scripts
    location = (0, 0, 0)
    rotation = (0, 0, 135)
    scale = (0.03, 0.03, 0.03)
    mesh = bt.readMesh(mesh_path, location, rotation, scale)
    bpy.ops.object.shade_smooth()
    
    # Load etree nodes and assigned nodes
    etree_nodes = load_file_as_ints(etree_path)
    assigned_nodes = load_file_as_ints(assigned_path)
    
    if len(etree_nodes) != len(assigned_nodes):
        print(f"Warning: File length mismatch! etree_nodes: {len(etree_nodes)}, assigned_nodes: {len(assigned_nodes)}")
    
    print(f"Loaded {len(etree_nodes)} etree nodes and {len(assigned_nodes)} assigned nodes")
    
    num_separators = apply_vertex_colors(mesh, etree_nodes, assigned_nodes, max_level)
    
    # Apply subdivision for smoother surface (before material)
    if subdivision > 0:
        bt.subdivision(mesh, level=subdivision)
    
    # Create and apply vertex color material
    create_vertex_color_material(mesh)
    
    print(f"Mesh has {len(mesh.data.vertices)} vertices, {num_separators} are separators")
    
    # Setup scene (same as render_mesh.py)
    bt.invisibleGround(shadowBrightness=0.9)
    
    camLocation = (3, 0, 2)
    lookAtLocation = (0, 0, 0.5)
    cam = bt.setCamera(camLocation, lookAtLocation, 45)
    
    bt.setLight_sun((6, -30, -155), 2, 0.3)
    bt.setLight_ambient(color=(0.1, 0.1, 0.1, 1))
    bt.shadowThreshold(alphaThreshold=0.05, interpolationMode='CARDINAL')
    
    # Create output directory
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save blend file
    blend_path = os.path.splitext(output_path)[0] + '.blend'
    bpy.ops.wm.save_mainfile(filepath=blend_path)
    print(f"Saved: {blend_path}")
    
    # Render
    bt.renderImage(output_path, cam)
    print(f"Rendered: {output_path}")


def main():
    args = parse_args()
    
    # Validate mesh path
    if not os.path.exists(args.mesh):
        print(f"Error: Mesh not found: {args.mesh}")
        sys.exit(1)
    
    # Generate default file paths if not specified
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_name = os.path.splitext(os.path.basename(args.mesh))[0]
    
    if args.etree is None:
        args.etree = os.path.join(script_dir, '..', 'output', 'render_data',
                                  f'PARTH_etree_nodes_{mesh_name}.txt')
    
    if args.assigned is None:
        args.assigned = os.path.join(script_dir, '..', 'output', 'render_data',
                                     f'PARTH_assigned_nodes_{mesh_name}.txt')
    
    if not os.path.exists(args.etree):
        print(f"Error: etree_nodes file not found: {args.etree}")
        sys.exit(1)
    
    if not os.path.exists(args.assigned):
        print(f"Error: assigned_nodes file not found: {args.assigned}")
        sys.exit(1)
    
    # Generate output path if not specified
    if args.output is None:
        figures_dir = os.path.join(script_dir, 'Figures')
        args.output = os.path.join(figures_dir, f'{mesh_name}_sep_{args.max_level}_color.png')
    
    print("=" * 60)
    print("Separator Vertex Color Visualization")
    print("=" * 60)
    print(f"Mesh: {args.mesh}")
    print(f"Etree Nodes: {args.etree}")
    print(f"Assigned Nodes: {args.assigned}")
    print(f"Output: {args.output}")
    print(f"Max Level: {args.max_level}")
    print(f"Subdivision: {args.subdivision}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Samples: {args.samples}")
    print("=" * 60)
    
    render_separators_colored(
        mesh_path=args.mesh,
        etree_path=args.etree,
        assigned_path=args.assigned,
        output_path=args.output,
        max_level=args.max_level,
        resolution=args.resolution,
        samples=args.samples,
        subdivision=args.subdivision
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
