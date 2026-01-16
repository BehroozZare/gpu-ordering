#!/usr/bin/env python3
"""
Separator Mesh Visualization using BlenderToolbox

Visualizes mesh partitioning based on elimination tree (etree) node labels.
Separators are shown as small spheres, partitions as colored submeshes.

Requires two data files:
    - PARTH_etree_nodes_{mesh}.txt: etree node IDs for each vertex
    - PARTH_assigned_nodes_{mesh}.txt: original vertex indices (mapping)

Usage:
    python show_separator_as_point.py --mesh /path/to/mesh.obj
    python show_separator_as_point.py --mesh /path/to/mesh.obj --etree /path/to/etree.txt --assigned /path/to/assigned.txt
"""

import bpy
import os
import sys
import argparse
import bmesh
from collections import defaultdict
import blendertoolbox as bt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize mesh separators')
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
    parser.add_argument('--separator_radius', type=float, default=0.008,
                        help='Radius of separator spheres')
    parser.add_argument('--resolution', type=int, default=1080,
                        help='Output resolution')
    parser.add_argument('--samples', type=int, default=200,
                        help='Number of render samples')
    parser.add_argument('--subdivision', type=int, default=1,
                        help='Subdivision level for smoother surface (0=none, 1=light, 2=smooth)')
    return parser.parse_args()


# Color presets (soft, pleasant colors for partitions)
COLOR_PRESETS = [
    (178/255, 223/255, 138/255, 1.0),  # Soft green
    (166/255, 206/255, 227/255, 1.0),  # Soft cyan
    (251/255, 180/255, 174/255, 1.0),  # Soft pink
    (253/255, 205/255, 172/255, 1.0),  # Soft peach
    (179/255, 179/255, 217/255, 1.0),  # Soft purple
    (252/255, 231/255, 149/255, 1.0),  # Soft yellow
    (203/255, 213/255, 232/255, 1.0),  # Soft steel blue
    (222/255, 203/255, 228/255, 1.0),  # Soft lilac
]

SEPARATOR_COLOR = (0.15, 0.15, 0.15, 1.0)  # Dark gray for separators


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


def get_partition_id(node_id, max_level):
    """Get partition ID for a non-separator node."""
    import math
    node_level = int(math.floor(math.log2(node_id + 1)))
    if node_level <= max_level:
        return -1
    
    # Walk up to find ancestor at level max_level + 1
    current = node_id
    current_level = node_level
    while current_level > max_level + 1:
        current = (current - 1) // 2
        current_level = int(math.floor(math.log2(current + 1)))
    
    start_node = (1 << (max_level + 1)) - 1
    return current - start_node


def separate_mesh(mesh_obj, etree_nodes, assigned_nodes, max_level):
    """
    Separate mesh into partitions and extract separator positions.
    Returns: (partition_meshes, separator_positions)
    """
    separator_verts = get_separator_vertices(etree_nodes, assigned_nodes, max_level)
    
    # Group non-separator vertices by partition
    # We iterate through both files to map original vertex -> partition
    partitions = defaultdict(list)
    for i, (etree_id, orig_vertex) in enumerate(zip(etree_nodes, assigned_nodes)):
        if orig_vertex not in separator_verts:
            pid = get_partition_id(etree_id, max_level)
            if pid >= 0:
                partitions[pid].append(orig_vertex)
    
    # Get separator world positions
    world_matrix = mesh_obj.matrix_world
    separator_positions = []
    for idx in separator_verts:
        if idx < len(mesh_obj.data.vertices):
            pos = world_matrix @ mesh_obj.data.vertices[idx].co
            separator_positions.append((pos.x, pos.y, pos.z))
    
    # Create partition submeshes
    partition_meshes = []
    for pid in sorted(partitions.keys()):
        keep_verts = set(partitions[pid])
        
        # Duplicate mesh
        bpy.ops.object.select_all(action='DESELECT')
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.object.duplicate()
        part_mesh = bpy.context.active_object
        part_mesh.name = f"Partition_{pid}"
        
        # Delete vertices not in partition
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(part_mesh.data)
        bm.verts.ensure_lookup_table()
        
        verts_to_delete = [v for v in bm.verts if v.index not in keep_verts]
        bmesh.ops.delete(bm, geom=verts_to_delete, context='VERTS')
        
        bmesh.update_edit_mesh(part_mesh.data)
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Clean loose geometry
        bpy.ops.object.select_all(action='DESELECT')
        part_mesh.select_set(True)
        bpy.context.view_layer.objects.active = part_mesh
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.delete_loose()
        bpy.ops.object.mode_set(mode='OBJECT')
        
        partition_meshes.append(part_mesh)
    
    # Hide original mesh
    mesh_obj.hide_set(True)
    mesh_obj.hide_render = True
    
    return partition_meshes, separator_positions


def create_separator_spheres(positions, radius, color):
    """Create instanced spheres at separator positions."""
    if not positions:
        return None
    
    # Create material
    mat = bpy.data.materials.new(name="SeparatorMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.inputs['Base Color'].default_value = color
    principled.inputs['Roughness'].default_value = 0.25
    
    output = nodes.new(type='ShaderNodeOutputMaterial')
    mat.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    # Create base sphere
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=8, ring_count=6)
    base_sphere = bpy.context.active_object
    base_sphere.name = "SeparatorSphereBase"
    base_sphere.data.materials.append(mat)
    
    # Create point cloud mesh
    point_mesh = bpy.data.meshes.new("SeparatorPoints")
    point_mesh.from_pydata(positions, [], [])
    point_obj = bpy.data.objects.new("SeparatorPointCloud", point_mesh)
    bpy.context.collection.objects.link(point_obj)
    
    # Setup geometry nodes for instancing
    geo_mod = point_obj.modifiers.new(name="InstanceSpheres", type='NODES')
    node_tree = bpy.data.node_groups.new(name="SphereInstancer", type='GeometryNodeTree')
    geo_mod.node_group = node_tree
    
    # Create nodes
    input_node = node_tree.nodes.new('NodeGroupInput')
    output_node = node_tree.nodes.new('NodeGroupOutput')
    instance_node = node_tree.nodes.new('GeometryNodeInstanceOnPoints')
    obj_info = node_tree.nodes.new('GeometryNodeObjectInfo')
    realize = node_tree.nodes.new('GeometryNodeRealizeInstances')
    
    obj_info.inputs['Object'].default_value = base_sphere
    
    # Add sockets (Blender 4.0+ API)
    node_tree.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    node_tree.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    
    # Link nodes
    links = node_tree.links
    links.new(input_node.outputs['Geometry'], instance_node.inputs['Points'])
    links.new(obj_info.outputs['Geometry'], instance_node.inputs['Instance'])
    links.new(instance_node.outputs['Instances'], realize.inputs['Geometry'])
    links.new(realize.outputs['Geometry'], output_node.inputs['Geometry'])
    
    # Hide base sphere
    base_sphere.hide_set(True)
    base_sphere.hide_render = True
    
    return point_obj


def render_separators(mesh_path, etree_path, assigned_path, output_path, max_level=0,
                      separator_radius=0.008, resolution=1080, samples=200,
                      subdivision=1):
    """Main rendering function."""
    
    # Initialize Blender (same as render_mesh.py)
    bt.blenderInit(resolution, resolution, samples, 1.5)
    
    # Load mesh with same transforms as render_mesh.py
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
    
    partition_meshes, separator_positions = separate_mesh(mesh, etree_nodes, assigned_nodes, max_level)
    
    print(f"Created {len(partition_meshes)} partitions, {len(separator_positions)} separator vertices")
    
    # Create separator spheres
    create_separator_spheres(separator_positions, separator_radius, SEPARATOR_COLOR)
    
    # Apply materials and subdivision to partitions (like render_mesh.py)
    for i, part_mesh in enumerate(partition_meshes):
        # Subdivision for smoother surface
        if subdivision > 0:
            bt.subdivision(part_mesh, level=subdivision)
        # Balloon material for soft diffuse look
        color = COLOR_PRESETS[i % len(COLOR_PRESETS)]
        mesh_color = bt.colorObj(color, 0.5, 1.0, 1.0, 0.0, 2.0)
        bt.setMat_balloon(part_mesh, mesh_color, 0.0)
    
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
        args.output = os.path.join(figures_dir, f'{mesh_name}_sep_{args.max_level}_point.png')
    
    print("=" * 60)
    print("Separator Mesh Visualization")
    print("=" * 60)
    print(f"Mesh: {args.mesh}")
    print(f"Etree Nodes: {args.etree}")
    print(f"Assigned Nodes: {args.assigned}")
    print(f"Output: {args.output}")
    print(f"Max Level: {args.max_level}")
    print(f"Separator Radius: {args.separator_radius}")
    print(f"Subdivision: {args.subdivision}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Samples: {args.samples}")
    print("=" * 60)
    
    render_separators(
        mesh_path=args.mesh,
        etree_path=args.etree,
        assigned_path=args.assigned,
        output_path=args.output,
        max_level=args.max_level,
        separator_radius=args.separator_radius,
        resolution=args.resolution,
        samples=args.samples,
        subdivision=args.subdivision
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
