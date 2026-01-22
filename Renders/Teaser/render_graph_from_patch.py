#!/usr/bin/env python3
"""
Patch Graph Visualization using BlenderToolbox

Renders a graph where each patch is a node (sphere) and edges connect
adjacent patches. Nodes are positioned at patch centroids in 3D space.

Requires:
    - pip install bpy==4.3.0 --extra-index-url https://download.blender.org/pypi/
    - pip install blendertoolbox

Usage:
    python render_graph_from_patch.py --mesh /path/to/mesh.obj --patch_data /path/to/patches.txt
    python render_graph_from_patch.py --help
"""

import bpy
import os
import sys
import argparse
import math
from collections import defaultdict
from mathutils import Vector

import blendertoolbox as bt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Render patch adjacency graph')
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
    parser.add_argument('--node_scale', type=float, default=1.0,
                        help='Scale factor for node sphere radius')
    parser.add_argument('--edge_scale', type=float, default=1.0,
                        help='Scale factor for edge cylinder radius')
    parser.add_argument('--show_mesh', action='store_true',
                        help='Show the original mesh as a transparent overlay')
    return parser.parse_args()


# High-contrast colors for patches (same as render_patch.py)
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

# Edge color (dark gray)
EDGE_COLOR = (0.3, 0.3, 0.3, 1.0)


def load_patch_ids(filepath):
    """Load patch IDs from file (one integer per line)."""
    with open(filepath, 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]


def compute_patch_centroids(mesh_obj, patch_ids):
    """
    Compute the centroid (center of mass) for each patch.
    
    Args:
        mesh_obj: Blender mesh object
        patch_ids: List of patch IDs (index i = patch ID for vertex i)
    
    Returns:
        Dictionary mapping patch ID to centroid Vector (world coordinates)
    """
    vertices = mesh_obj.data.vertices
    patch_vertices = defaultdict(list)
    
    # Group vertices by patch
    num_vertices = min(len(patch_ids), len(vertices))
    for i in range(num_vertices):
        pid = patch_ids[i]
        # Transform vertex to world coordinates
        world_pos = mesh_obj.matrix_world @ vertices[i].co
        patch_vertices[pid].append(world_pos)
    
    # Compute centroid for each patch
    centroids = {}
    for pid, verts in patch_vertices.items():
        centroid = Vector((0, 0, 0))
        for v in verts:
            centroid += v
        centroids[pid] = centroid / len(verts)
    
    return centroids


def compute_patch_adjacency(mesh_obj, patch_ids):
    """
    Compute which patches are adjacent (share at least one edge).
    
    Args:
        mesh_obj: Blender mesh object
        patch_ids: List of patch IDs (index i = patch ID for vertex i)
    
    Returns:
        Set of tuples (patch_id_1, patch_id_2) where patch_id_1 < patch_id_2
    """
    adjacency = set()
    num_vertices = len(patch_ids)
    
    for edge in mesh_obj.data.edges:
        v1, v2 = edge.vertices
        # Skip if vertex indices are out of range
        if v1 >= num_vertices or v2 >= num_vertices:
            continue
        
        p1, p2 = patch_ids[v1], patch_ids[v2]
        if p1 != p2:
            # Store as ordered pair to avoid duplicates
            adjacency.add((min(p1, p2), max(p1, p2)))
    
    return adjacency


def create_solid_color_material(color, name="Material"):
    """
    Create a material with a solid color.
    
    Args:
        color: RGBA tuple
        name: Material name
    
    Returns:
        Blender material
    """
    mat = bpy.data.materials.new(name=name)
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
    
    # Principled BSDF
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (100, 0)
    
    # Set color and material properties
    principled.inputs['Base Color'].default_value = color
    if 'Roughness' in principled.inputs:
        principled.inputs['Roughness'].default_value = 0.4
    if 'Metallic' in principled.inputs:
        principled.inputs['Metallic'].default_value = 0.0
    if 'Specular IOR Level' in principled.inputs:
        principled.inputs['Specular IOR Level'].default_value = 0.3
    
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    return mat


def create_node_sphere(location, radius, color, name):
    """
    Create a sphere at the given location to represent a graph node.
    
    Args:
        location: Vector or tuple for sphere center
        radius: Sphere radius
        color: RGBA color tuple
        name: Object name
    
    Returns:
        Blender sphere object
    """
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        segments=32,
        ring_count=16,
        location=location
    )
    sphere = bpy.context.active_object
    sphere.name = name
    
    # Smooth shading
    bpy.ops.object.shade_smooth()
    
    # Apply material
    mat = create_solid_color_material(color, f"{name}_mat")
    sphere.data.materials.append(mat)
    
    return sphere


def create_edge_cylinder(start, end, radius, color, name):
    """
    Create a cylinder connecting two points (graph edge).
    
    Args:
        start: Start point Vector
        end: End point Vector
        radius: Cylinder radius
        color: RGBA color tuple
        name: Object name
    
    Returns:
        Blender cylinder object
    """
    # Calculate midpoint, direction, and length
    direction = end - start
    length = direction.length
    
    if length < 1e-6:
        return None  # Skip degenerate edges
    
    midpoint = (start + end) / 2
    
    # Create cylinder at origin first
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius,
        depth=length,
        vertices=16,
        location=(0, 0, 0)
    )
    cylinder = bpy.context.active_object
    cylinder.name = name
    
    # Calculate rotation to align cylinder with direction
    # Default cylinder is along Z axis
    up = Vector((0, 0, 1))
    rotation_quat = up.rotation_difference(direction)
    cylinder.rotation_euler = rotation_quat.to_euler()
    
    # Move to midpoint
    cylinder.location = midpoint
    
    # Smooth shading
    bpy.ops.object.shade_smooth()
    
    # Apply material
    mat = create_solid_color_material(color, f"{name}_mat")
    cylinder.data.materials.append(mat)
    
    return cylinder


def compute_auto_scale(centroids, num_patches):
    """
    Compute automatic scale factors for nodes and edges based on mesh size.
    
    Args:
        centroids: Dictionary of patch centroids
        num_patches: Number of patches
    
    Returns:
        Tuple (node_radius, edge_radius)
    """
    if len(centroids) < 2:
        return 0.05, 0.01
    
    # Compute bounding box of centroids
    positions = list(centroids.values())
    min_pos = Vector((
        min(p.x for p in positions),
        min(p.y for p in positions),
        min(p.z for p in positions)
    ))
    max_pos = Vector((
        max(p.x for p in positions),
        max(p.y for p in positions),
        max(p.z for p in positions)
    ))
    
    bbox_size = (max_pos - min_pos).length
    
    # Scale based on number of patches and bounding box
    node_radius = bbox_size / (math.sqrt(num_patches) * 4)
    edge_radius = node_radius * 0.3
    
    # Clamp to reasonable values
    node_radius = max(0.01, min(0.2, node_radius))
    edge_radius = max(0.003, min(0.05, edge_radius))
    
    return node_radius, edge_radius


def render_patch_graph(mesh_path, patch_data_path, output_path, resolution=1080,
                       samples=200, node_scale=1.0, edge_scale=1.0, show_mesh=False):
    """
    Main rendering function for patch graph visualization.
    
    Args:
        mesh_path: Path to mesh file
        patch_data_path: Path to vertex-to-patch mapping file
        output_path: Output image path
        resolution: Output resolution (square)
        samples: Render samples
        node_scale: Scale factor for node radius
        edge_scale: Scale factor for edge radius
        show_mesh: Whether to show original mesh as overlay
    """
    # Initialize Blender (same settings as render_patch.py)
    bt.blenderInit(resolution, resolution, samples, 1.5)
    
    # Load mesh with same transforms as render_patch.py
    location = (0, 0.09, 0.68)
    mesh_scale = (2, 2, 2)
    rotation = (90, 0, 50)
    mesh = bt.readMesh(mesh_path, location, rotation, mesh_scale)
    
    # Load patch data
    print(f"Loading patch data from: {patch_data_path}")
    patch_ids = load_patch_ids(patch_data_path)
    print(f"Loaded {len(patch_ids)} patch assignments")
    
    # Compute patch centroids
    print("Computing patch centroids...")
    centroids = compute_patch_centroids(mesh, patch_ids)
    num_patches = len(centroids)
    print(f"Found {num_patches} patches")
    
    # Compute patch adjacency
    print("Computing patch adjacency...")
    adjacency = compute_patch_adjacency(mesh, patch_ids)
    print(f"Found {len(adjacency)} edges between patches")
    
    # Compute auto scale
    base_node_radius, base_edge_radius = compute_auto_scale(centroids, num_patches)
    node_radius = base_node_radius * node_scale
    edge_radius = base_edge_radius * edge_scale
    print(f"Node radius: {node_radius:.4f}, Edge radius: {edge_radius:.4f}")
    
    # Hide or delete the original mesh (we only want the graph)
    if not show_mesh:
        bpy.data.objects.remove(mesh, do_unlink=True)
    else:
        # Make mesh transparent/wireframe
        mesh.display_type = 'WIRE'
    
    # Create graph nodes (spheres)
    print("Creating graph nodes...")
    node_objects = {}
    for pid, centroid in centroids.items():
        color = PATCH_COLORS[pid % len(PATCH_COLORS)]
        sphere = create_node_sphere(
            location=centroid,
            radius=node_radius,
            color=color,
            name=f"Node_{pid}"
        )
        node_objects[pid] = sphere
    
    # Create graph edges (cylinders)
    print("Creating graph edges...")
    edge_objects = []
    for p1, p2 in adjacency:
        start = centroids[p1]
        end = centroids[p2]
        cylinder = create_edge_cylinder(
            start=start,
            end=end,
            radius=edge_radius,
            color=EDGE_COLOR,
            name=f"Edge_{p1}_{p2}"
        )
        if cylinder:
            edge_objects.append(cylinder)
    
    print(f"Created {len(node_objects)} nodes and {len(edge_objects)} edges")
    
    # Setup scene (same as render_patch.py)
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
        args.output = os.path.join(figures_dir, f'{mesh_name}_graph.png')
    
    print("=" * 60)
    print("Patch Graph Visualization")
    print("=" * 60)
    print(f"Mesh: {args.mesh}")
    print(f"Patch Data: {args.patch_data}")
    print(f"Output: {args.output}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Samples: {args.samples}")
    print(f"Node Scale: {args.node_scale}")
    print(f"Edge Scale: {args.edge_scale}")
    print(f"Show Mesh: {args.show_mesh}")
    print("=" * 60)
    
    render_patch_graph(
        mesh_path=args.mesh,
        patch_data_path=args.patch_data,
        output_path=args.output,
        resolution=args.resolution,
        samples=args.samples,
        node_scale=args.node_scale,
        edge_scale=args.edge_scale,
        show_mesh=args.show_mesh
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
