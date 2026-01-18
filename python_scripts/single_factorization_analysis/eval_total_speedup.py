#!/usr/bin/env python3
"""
Speedup Analysis Script

Computes and visualizes speedup of PATCH_ORDERING (best configuration) over DEFAULT
ordering for CUDSS and MKL solvers.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 12


def compute_speedup_for_solver(df, solver_type):
    """Compute speedup data for a given solver type."""
    # Filter for DEFAULT ordering_type and specified solver_type
    default_df = df[
        (df["ordering_type"] == "DEFAULT") & 
        (df["solver_type"] == solver_type)
    ].copy()
    
    # Filter for PATCH_ORDERING and specified solver_type (all nd_levels)
    patch_df = df[
        (df["ordering_type"] == "PATCH_ORDERING") & 
        (df["solver_type"] == solver_type)
    ].copy()
    
    print(f"DEFAULT entries ({solver_type}): {len(default_df)}")
    print(f"PATCH_ORDERING ({solver_type}) entries: {len(patch_df)}")
    
    # Compute total runtime for DEFAULT
    # Treat ordering_time = -1 as 0
    default_df["total_time"] = (
        default_df["ordering_integration_time"].clip(lower=0) +
        default_df["analysis_time"] +
        default_df["factorization_time"] +
        default_df["solve_time"]
    )
    
    # Compute total runtime for PATCH_ORDERING
    # ordering_time captures the full ordering computation time (includes patch creation)
    # ordering_integration_time is the time to integrate/apply the ordering to the solver
    patch_df["total_time"] = (
        patch_df["patch_time"].clip(lower=0) +
        patch_df["ordering_time"].clip(lower=0) +
        patch_df["ordering_integration_time"].clip(lower=0) +
        patch_df["analysis_time"] +
        patch_df["factorization_time"] +
        patch_df["solve_time"]
    )
    
    # For each mesh_name, keep only the entry with the smallest total_time
    patch_df = patch_df.loc[patch_df.groupby("mesh_name")["total_time"].idxmin()]
    print(f"PATCH_ORDERING (best config, {solver_type}) entries: {len(patch_df)}")
    
    # Prepare dataframes for merging - keep only necessary columns
    default_for_merge = default_df[["mesh_name", "G_N", "total_time"]].rename(
        columns={"total_time": "default_total_time"}
    )
    patch_for_merge = patch_df[["mesh_name", "total_time"]].rename(
        columns={"total_time": "patch_total_time"}
    )
    
    # Merge on mesh_name to pair corresponding entries
    merged_df = pd.merge(default_for_merge, patch_for_merge, on="mesh_name", how="inner")
    print(f"Matched mesh pairs ({solver_type}): {len(merged_df)}")
    
    # Compute speedup: DEFAULT / PATCH_ORDERING
    merged_df["speedup"] = merged_df["default_total_time"] / merged_df["patch_total_time"]
    
    # Sort by G_N for a cleaner plot
    merged_df = merged_df.sort_values("G_N")
    
    return merged_df


def plot_speedup(ax, merged_df, solver_type, color='C0'):
    """Plot speedup data on the given axes."""
    ax.scatter(merged_df["G_N"], merged_df["speedup"], 
               alpha=0.7, edgecolors='black', linewidth=0.5, color=color)
    
    # Use log scale for x-axis due to wide range of G_N values
    ax.set_xscale("log")
    
    # Add horizontal line at speedup = 1 (no speedup)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='No speedup')
    
    # Remove upper and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_ylabel(f"Speedup ({solver_type})")
    
    ax.grid(True, alpha=0.3)
    
    # Add some statistics as text
    mean_speedup = merged_df["speedup"].mean()
    min_speedup = merged_df["speedup"].min()
    max_speedup = merged_df["speedup"].max()
    
    stats_text = f"Mean: {mean_speedup:.2f}x\nMin: {min_speedup:.2f}x\nMax: {max_speedup:.2f}x"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def main():
    # Get the path to the data file
    script_dir = Path(__file__).parent
    data_file = script_dir / "data" / "laplace_full_benchmark.csv"
    result_dir = script_dir / "results"
    
    # Read the CSV file
    df = pd.read_csv(data_file)
    
    print(f"Total entries: {len(df)}")
    
    # Compute speedup for both solvers
    cudss_df = compute_speedup_for_solver(df, "CUDSS")
    mkl_df = compute_speedup_for_solver(df, "MKL")
    
    # Create the plot with two subplots stacked vertically
    scale_factor = 2
    fig, (ax_cudss, ax_mkl) = plt.subplots(2, 1, figsize=(3.36 * scale_factor, 1.5 * scale_factor * 2))
    
    # Plot CUDSS on top
    plot_speedup(ax_cudss, cudss_df, "CUDSS")
    
    # Plot MKL on bottom (red color)
    plot_speedup(ax_mkl, mkl_df, "MKL", color='red')
    ax_mkl.set_xlabel(r"Number of mesh vertices")
    
    plt.tight_layout()
    
    # Save the figure
    output_path = result_dir / "single_factor_total_speedup.pdf"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
