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
import math
import blendertoolbox as bt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize mesh separators')
    parser.add_argument('--mesh', type=str,
                        default='/media/behrooz/FarazHard/Last_Project/BenchmarkMesh/tri-mesh/final/beetle.obj',
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
    parser.add_argument('--resolution', type=int, default=2048,
                        help='Output resolution')
    parser.add_argument('--samples', type=int, default=500,
                        help='Number of render samples')
    parser.add_argument('--subdivision', type=int, default=1,
                        help='Subdivision level for smoother surface (0=none, 1=light, 2=smooth)')
    return parser.parse_args()


# Color presets (distinct, high-contrast colors for partitions)
# Based on a qualitative palette (Tableau-like) for better separation at higher levels.
COLOR_PRESETS = [
    (31/255, 119/255, 180/255, 1.0),   # Blue
    (255/255, 127/255, 14/255, 1.0),   # Orange
    (44/255, 160/255, 44/255, 1.0),    # Green
    (214/255, 39/255, 40/255, 1.0),    # Red
    (148/255, 103/255, 189/255, 1.0),  # Purple
    (140/255, 86/255, 75/255, 1.0),    # Brown
    (227/255, 119/255, 194/255, 1.0),  # Pink
    (188/255, 189/255, 34/255, 1.0),   # Olive / Yellow-green
    (23/255, 190/255, 207/255, 1.0),   # Cyan
    (255/255, 152/255, 150/255, 1.0),  # Light red
    (197/255, 176/255, 213/255, 1.0),  # Light purple
    (196/255, 156/255, 148/255, 1.0),  # Light brown
]

# Separator colors per etree level (0=root, 1=children, ...)
# - level 0: near-black (or swap to red if you prefer)
# - level 1: blue
# - higher levels: distinct colors cycling
SEPARATOR_LEVEL_COLORS = [
    (0.08, 0.08, 0.08, 1.0),  # level 0: near-black
    (0.20, 0.45, 0.95, 1.0),  # level 1: blue
    (0.90, 0.20, 0.20, 1.0),  # level 2: red
    (0.20, 0.80, 0.30, 1.0),  # level 3: green
    (0.90, 0.60, 0.10, 1.0),  # level 4: orange
    (0.60, 0.30, 0.90, 1.0),  # level 5: purple
]


def force_object_materials_opaque(obj):
    """
    Force all materials used by `obj` to render opaque.

    This is a defensive post-process step intended to override any semi-transparent
    settings that may be introduced by BlenderToolbox materials.
    """
    if obj is None or getattr(obj, "data", None) is None:
        return

    mats = getattr(obj.data, "materials", None)
    if not mats:
        return

    for mat in mats:
        if mat is None:
            continue

        # Viewport/render blending
        if hasattr(mat, "blend_method"):
            mat.blend_method = 'OPAQUE'
        if hasattr(mat, "shadow_method"):
            mat.shadow_method = 'OPAQUE'

        # Some Blender APIs still consult diffuse_color alpha
        try:
            if hasattr(mat, "diffuse_color") and len(mat.diffuse_color) >= 4:
                mat.diffuse_color[3] = 1.0
        except Exception:
            pass

        # Node-based materials: ensure Principled is fully opaque
        if not getattr(mat, "use_nodes", False) or mat.node_tree is None:
            continue

        for node in mat.node_tree.nodes:
            if getattr(node, "type", None) != 'BSDF_PRINCIPLED':
                continue

            # Alpha -> 1.0
            for alpha_name in ("Alpha",):
                sock = node.inputs.get(alpha_name) if hasattr(node.inputs, "get") else None
                if sock is not None and hasattr(sock, "default_value"):
                    try:
                        sock.default_value = 1.0
                    except Exception:
                        pass

            # Transmission -> 0.0 (Blender versions differ on socket naming)
            for trans_name in ("Transmission", "Transmission Weight"):
                sock = node.inputs.get(trans_name) if hasattr(node.inputs, "get") else None
                if sock is not None and hasattr(sock, "default_value"):
                    try:
                        sock.default_value = 0.0
                    except Exception:
                        pass


def set_object_opaque_principled_material(obj, base_color_rgba, name="OpaquePartitionMaterial"):
    """
    Replace all material slots on `obj` with a guaranteed-opaque Principled material.

    This is stronger than `force_object_materials_opaque()` and is used when upstream
    materials include transparency nodes (e.g., Transparent BSDF / Mix Shader) that
    are hard to override reliably.
    """
    if obj is None or getattr(obj, "data", None) is None:
        return

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    # Ensure opaque render settings
    if hasattr(mat, "blend_method"):
        mat.blend_method = 'OPAQUE'
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = 'OPAQUE'
    try:
        if hasattr(mat, "diffuse_color") and len(mat.diffuse_color) >= 4:
            mat.diffuse_color = (base_color_rgba[0], base_color_rgba[1], base_color_rgba[2], 1.0)
    except Exception:
        pass

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new(type='ShaderNodeOutputMaterial')
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')

    # Color + "soft balloon-ish" look, but fully opaque
    principled.inputs['Base Color'].default_value = (
        float(base_color_rgba[0]),
        float(base_color_rgba[1]),
        float(base_color_rgba[2]),
        1.0,
    )
    if 'Roughness' in principled.inputs:
        principled.inputs['Roughness'].default_value = 0.45
    if 'Metallic' in principled.inputs:
        principled.inputs['Metallic'].default_value = 0.0
    if 'Alpha' in principled.inputs:
        principled.inputs['Alpha'].default_value = 1.0
    for trans_name in ('Transmission', 'Transmission Weight'):
        if trans_name in principled.inputs:
            principled.inputs[trans_name].default_value = 0.0
    if 'Specular IOR Level' in principled.inputs:
        principled.inputs['Specular IOR Level'].default_value = 0.25

    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    # Replace all slots to avoid leaving any transparent materials behind
    mats = getattr(obj.data, "materials", None)
    if mats is None:
        return
    if len(mats) == 0:
        mats.append(mat)
    else:
        for idx in range(len(mats)):
            mats[idx] = mat


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


def get_separator_vertices_by_level(etree_nodes, assigned_nodes, max_level):
    """
    Group separator vertices by etree level.

    For a given `max_level`, separators are all entries with:
        etree_id <= (1 << (max_level + 1)) - 2

    The per-separator "level" is computed as:
        level(etree_id) = floor(log2(etree_id + 1))
    so etree_id=0 -> level 0, etree_id in {1,2} -> level 1, etc.
    """
    max_etree_id = (1 << (max_level + 1)) - 2
    by_level = defaultdict(set)

    for etree_id, orig_vertex in zip(etree_nodes, assigned_nodes):
        if etree_id <= max_etree_id:
            level = int(math.floor(math.log2(etree_id + 1)))
            by_level[level].add(orig_vertex)

    return dict(by_level)


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
    Returns: (partition_meshes, separator_positions_by_level)
    """
    separator_verts_by_level = get_separator_vertices_by_level(etree_nodes, assigned_nodes, max_level)
    separator_verts = set()
    for verts in separator_verts_by_level.values():
        separator_verts.update(verts)
    
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
    separator_positions_by_level = {}
    for level, verts in separator_verts_by_level.items():
        positions = []
        for idx in verts:
            if idx < len(mesh_obj.data.vertices):
                pos = world_matrix @ mesh_obj.data.vertices[idx].co
                positions.append((pos.x, pos.y, pos.z))
        separator_positions_by_level[level] = positions
    
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
    
    return partition_meshes, separator_positions_by_level


def create_separator_spheres(positions, radius, color, name_suffix=""):
    """Create instanced spheres at separator positions."""
    if not positions:
        return None

    suffix = f"_{name_suffix}" if name_suffix else ""
    
    # Create material
    mat = bpy.data.materials.new(name=f"SeparatorMaterial{suffix}")
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
    base_sphere.name = f"SeparatorSphereBase{suffix}"
    base_sphere.data.materials.append(mat)
    
    # Create point cloud mesh
    point_mesh = bpy.data.meshes.new(f"SeparatorPoints{suffix}")
    point_mesh.from_pydata(positions, [], [])
    point_obj = bpy.data.objects.new(f"SeparatorPointCloud{suffix}", point_mesh)
    bpy.context.collection.objects.link(point_obj)
    
    # Setup geometry nodes for instancing
    geo_mod = point_obj.modifiers.new(name=f"InstanceSpheres{suffix}", type='NODES')
    node_tree = bpy.data.node_groups.new(name=f"SphereInstancer{suffix}", type='GeometryNodeTree')
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
    location = (0, 0.09, 0.68)
    rotation = (90, 0, 50)
    mesh_scale = (2, 2, 2)
    mesh = bt.readMesh(mesh_path, location, rotation, mesh_scale)
    bpy.ops.object.shade_smooth()
    
    # Load etree nodes and assigned nodes
    etree_nodes = load_file_as_ints(etree_path)
    assigned_nodes = load_file_as_ints(assigned_path)
    
    if len(etree_nodes) != len(assigned_nodes):
        print(f"Warning: File length mismatch! etree_nodes: {len(etree_nodes)}, assigned_nodes: {len(assigned_nodes)}")
    
    print(f"Loaded {len(etree_nodes)} etree nodes and {len(assigned_nodes)} assigned nodes")
    
    partition_meshes, separator_positions_by_level = separate_mesh(mesh, etree_nodes, assigned_nodes, max_level)
    
    total_seps = sum(len(v) for v in separator_positions_by_level.values())
    print(f"Created {len(partition_meshes)} partitions, {total_seps} separator vertices across {len(separator_positions_by_level)} levels")
    
    # Create separator spheres per level (distinct colors)
    for level in sorted(separator_positions_by_level.keys()):
        positions = separator_positions_by_level[level]
        color = SEPARATOR_LEVEL_COLORS[level % len(SEPARATOR_LEVEL_COLORS)]
        create_separator_spheres(
            positions,
            separator_radius,
            color,
            name_suffix=f"L{level}"
        )
    
    # Apply materials and subdivision to partitions (like render_mesh.py)
    for i, part_mesh in enumerate(partition_meshes):
        # Subdivision for smoother surface
        if subdivision > 0:
            bt.subdivision(part_mesh, level=subdivision)
        # Opaque Principled material (keeps partition colors but removes transparency)
        color = COLOR_PRESETS[i % len(COLOR_PRESETS)]
        set_object_opaque_principled_material(
            part_mesh,
            base_color_rgba=color,
            name=f"PartitionOpaque_{i}"
        )
    
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
