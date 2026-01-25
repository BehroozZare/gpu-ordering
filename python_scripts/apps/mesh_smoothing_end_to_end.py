#!/usr/bin/env python3
"""
Mesh Smoothing End-to-End Speedup Analysis

Computes and visualizes end-to-end speedup of PATCH_ORDERING over DEFAULT
ordering for the mesh smoothing benchmark (cuDSS solver).

Usage:
    python mesh_smoothing_end_to_end.py              # Use all iterations (default)
    python mesh_smoothing_end_to_end.py --num-iter 3 # Use only first 3 iterations (0, 1, 2)
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

mpl.rcParams['font.family'] = ['TeX Gyre Pagella', 'serif']
mpl.rcParams['font.size'] = 18


def compute_total_time_default(row):
    """
    Compute total time for DEFAULT ordering.
    Formula: ordering_integration_time + analysis_time + factorization_time + solve_time
    """
    return (
        max(0, row["ordering_integration_time"]) +
        row["analysis_time"] +
        row["factorization_time"] +
        row["solve_time"]
    )


def compute_total_time_patch(row):
    """
    Compute total time for PATCH_ORDERING.
    Formula: patch_time (only at iter 0) + ordering_time + ordering_integration_time + 
             analysis_time + factorization_time + solve_time
    """
    # patch_time is stored as constant in all rows, but should only count at iteration 0
    patch_time = max(0, row["patch_time"]) if row["iteration"] == 0 else 0
    
    return (
        patch_time +
        max(0, row["ordering_time"]) +
        max(0, row["ordering_integration_time"]) +
        row["analysis_time"] +
        row["factorization_time"] +
        row["solve_time"]
    )


def compute_speedup_per_mesh(df, num_iter=None):
    """
    Compute end-to-end speedup for each mesh.
    
    Args:
        df: DataFrame with benchmark data
        num_iter: Number of iterations to include (None = all iterations)
                  e.g., num_iter=3 means iterations 0, 1, 2
    
    Returns:
        DataFrame with mesh_name, G_N, speedup
    """
    results = []
    
    mesh_names = df["mesh_name"].unique()
    
    for mesh_name in mesh_names:
        mesh_df = df[df["mesh_name"] == mesh_name].copy()
        
        # Filter by iteration count if specified
        if num_iter is not None:
            mesh_df = mesh_df[mesh_df["iteration"] < num_iter]
        
        # Separate DEFAULT and PATCH_ORDERING
        default_df = mesh_df[mesh_df["ordering_type"] == "DEFAULT"].copy()
        patch_df = mesh_df[mesh_df["ordering_type"] == "PATCH_ORDERING"].copy()
        
        if default_df.empty or patch_df.empty:
            print(f"  Warning: Missing data for mesh {mesh_name}")
            continue
        
        # Compute total_time for each row
        default_df["total_time"] = default_df.apply(compute_total_time_default, axis=1)
        patch_df["total_time"] = patch_df.apply(compute_total_time_patch, axis=1)
        
        # Speedup: sum all iterations
        default_total = default_df["total_time"].sum()
        patch_total = patch_df["total_time"].sum()
        speedup = default_total / patch_total if patch_total > 0 else float('inf')
        
        # Get mesh size (G_N)
        G_N = mesh_df["G_N"].iloc[0]
        
        results.append({
            "mesh_name": mesh_name,
            "G_N": G_N,
            "speedup": speedup,
        })
    
    return pd.DataFrame(results)


def plot_speedup(ax, speedup_df):
    """Plot speedup scatter plot."""
    ax.scatter(speedup_df["G_N"], speedup_df["speedup"], 
               alpha=0.7, edgecolors='black', linewidth=0.2, color='C0', s=120)
    
    # Use log scale for x-axis due to wide range of G_N values
    ax.set_xscale("log")
    
    # Add horizontal line at speedup = 1 (no speedup)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='No speedup')
    
    # Remove upper and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlabel("Number of mesh vertices")
    ax.set_ylabel("Speedup (cuDSS)")
    
    ax.grid(True, alpha=0.3)
    
    # Add statistics as text
    mean_speedup = speedup_df["speedup"].mean()
    min_speedup = speedup_df["speedup"].min()
    max_speedup = speedup_df["speedup"].max()
    
    stats_text = f"Avg: {mean_speedup:.2f}x\nMin: {min_speedup:.2f}x\nMax: {max_speedup:.2f}x"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Compute and plot end-to-end speedup for mesh smoothing benchmark"
    )
    parser.add_argument(
        "--num-iter", "-n",
        type=int,
        default=None,
        help="Number of iterations to include (default: all). "
             "E.g., --num-iter 3 uses iterations 0, 1, 2"
    )
    args = parser.parse_args()
    
    # Get the path to the data file
    script_dir = Path(__file__).parent
    data_file = script_dir / "data" / "Smoothing" / "smoothing.csv"
    result_dir = script_dir / "results"
    
    # Create results directory if it doesn't exist
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(data_file)
    
    print(f"Total entries: {len(df)}")
    print(f"Unique meshes: {df['mesh_name'].nunique()}")
    
    # Filter for CUDSS solver only
    df = df[df["solver_type"] == "CUDSS"]
    print(f"CUDSS entries: {len(df)}")
    
    # Report iteration setting
    if args.num_iter is not None:
        print(f"Using first {args.num_iter} iteration(s): 0 to {args.num_iter - 1}")
    else:
        max_iter = df["iteration"].max() + 1
        print(f"Using all {max_iter} iterations")
    
    # Compute speedup for each mesh
    speedup_df = compute_speedup_per_mesh(df, num_iter=args.num_iter)
    speedup_df = speedup_df.sort_values("G_N")
    
    print(f"\nSpeedup results ({len(speedup_df)} meshes):")
    print(speedup_df.to_string(index=False))
    
    # Create the plot
    scale_factor = 2
    fig, ax = plt.subplots(figsize=(3.36 * scale_factor, 1.5 * scale_factor))
    
    plot_speedup(ax, speedup_df)
    
    plt.tight_layout()
    
    # Save the figure (include num_iter in filename if specified)
    if args.num_iter is not None:
        output_path = result_dir / f"smoothing_end_to_end_speedup_iter{args.num_iter}.pdf"
    else:
        output_path = result_dir / "smoothing_end_to_end_speedup.pdf"
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
