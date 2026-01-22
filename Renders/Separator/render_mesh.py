#!/usr/bin/env python3
"""
Simple Mesh Renderer using BlenderToolbox

A clean script for rendering meshes beautifully with BlenderToolbox,
following the demo_balloon.py pattern from HTDerekLiu/BlenderToolbox.

Usage:
    python render_mesh.py --mesh /path/to/mesh.obj
    
Or use the run_render.sh wrapper script.

Requires:
    pip install bpy==4.3.0 --extra-index-url https://download.blender.org/pypi/
    pip install blendertoolbox
"""

import bpy
import os
import sys
import argparse
import blendertoolbox as bt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Render a mesh beautifully')
    parser.add_argument('--mesh', type=str, 
                        default='/media/behrooz/FarazHard/Last_Project/BenchmarkMesh/tri-mesh/final/beetle.obj',
                        help='Path to the mesh file (OBJ, PLY, etc.)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path (auto-generated if not specified)')
    parser.add_argument('--color', type=str, default='green',
                        choices=['green', 'blue', 'orange', 'pink', 'gray'],
                        help='Material color preset')
    parser.add_argument('--subdivision', type=int, default=1,
                        help='Subdivision level for smoother surface (0=none, 1=light, 2=smooth)')
    parser.add_argument('--resolution', type=int, default=1080,
                        help='Output resolution (square image)')
    parser.add_argument('--samples', type=int, default=200,
                        help='Number of render samples (higher=better quality)')
    parser.add_argument('--rotation', type=float, nargs=3, default=[90, 0, 120],
                        help='Mesh rotation (X Y Z) in degrees')
    parser.add_argument('--scale', type=float, default=1.5,
                        help='Mesh scale factor')
    
    return parser.parse_args()


# Color presets (soft, pleasant colors)
COLOR_PRESETS = {
    'green': (178/255, 223/255, 138/255, 1.0),   # Soft green (BlenderToolbox default)
    'blue': (166/255, 206/255, 227/255, 1.0),    # Soft sky blue
    'orange': (253/255, 205/255, 172/255, 1.0),  # Soft peach/orange
    'pink': (251/255, 180/255, 174/255, 1.0),    # Soft pink/coral
    'gray': (217/255, 217/255, 217/255, 1.0),    # Soft gray
}


def render_mesh(mesh_path, output_path, color_name='green', subdivision=1,
                resolution=1080, samples=200, rotation=(90, 0, 120), scale=0.03):
    """
    Render a mesh with beautiful BlenderToolbox styling.
    
    Args:
        mesh_path: Path to the mesh file
        output_path: Path for output image
        color_name: Color preset name
        subdivision: Subdivision level (0-2)
        resolution: Output resolution (square)
        samples: Render samples
        rotation: Mesh rotation tuple (degrees)
        scale: Mesh scale factor
    """
    # Get color from preset
    color_rgba = COLOR_PRESETS.get(color_name, COLOR_PRESETS['green'])
    
    # Initialize Blender
    imgRes_x = resolution
    imgRes_y = resolution
    numSamples = samples
    exposure = 1.5
    bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)
    
    # Load mesh
    location = (0, 0.09, 0.68)
    mesh_scale = (2, 2, 2)
    rotation = (90, 0, 50)
    mesh = bt.readMesh(mesh_path, location, rotation, mesh_scale)
    
    # Smooth shading
    bpy.ops.object.shade_smooth()
    
    # Subdivision for smoother surface (optional)
    if subdivision > 0:
        bt.subdivision(mesh, level=subdivision)
    
    # Set material - balloon material gives a nice soft, diffuse look
    # colorObj(RGBA, H, S, V, Bright, Contrast)
    meshColor = bt.colorObj(color_rgba, 0.5, 1.0, 1.0, 0.0, 2.0)
    AOStrength = 0.0
    bt.setMat_balloon(mesh, meshColor, AOStrength)
    
    # Set invisible ground (shadow catcher)
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
    
    # Generate output path if not specified
    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        figures_dir = os.path.join(script_dir, 'Figures')
        mesh_name = os.path.splitext(os.path.basename(args.mesh))[0]
        output_path = os.path.join(figures_dir, f'{mesh_name}.png')
    else:
        output_path = args.output
    
    print("=" * 60)
    print("Simple Mesh Renderer")
    print("=" * 60)
    print(f"Mesh: {args.mesh}")
    print(f"Output: {output_path}")
    print(f"Color: {args.color}")
    print(f"Subdivision: {args.subdivision}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Samples: {args.samples}")
    print(f"Rotation: {args.rotation}")
    print(f"Scale: {args.scale}")
    print("=" * 60)
    
    render_mesh(
        mesh_path=args.mesh,
        output_path=output_path,
        color_name=args.color,
        subdivision=args.subdivision,
        resolution=args.resolution,
        samples=args.samples,
        rotation=tuple(args.rotation),
        scale=args.scale
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
