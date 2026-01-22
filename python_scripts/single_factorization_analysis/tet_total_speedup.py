#!/usr/bin/env python3
"""
Tet Mesh Speedup Analysis Script

Computes and visualizes speedup of PATCH_ORDERING (best configuration) over DEFAULT
ordering for CUDSS and MKL solvers on tetrahedral mesh data.

Usage:
    python tet_total_speedup.py                  # Exclude patch_time (default)
    python tet_total_speedup.py --include-patch-time  # Include patch_time
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

# mpl.rcParams['font.family'] = ['Palatino Linotype', 'TeX Gyre Pagella', 'serif']
mpl.rcParams['font.family'] = ['TeX Gyre Pagella', 'serif']
mpl.rcParams['font.size'] = 18

# mpl.rcParams['pdf.fonttype'] = 42
# mpl.rcParams['ps.fonttype'] = 42
# mpl.rcParams['font.size'] = 12


def compute_speedup_for_solver(df, solver_type, include_patch_time=False):
    """Compute speedup data for a given solver type.
    
    Args:
        df: DataFrame with benchmark data
        solver_type: "CUDSS" or "MKL"
        include_patch_time: If True, include patch_time in PATCH_ORDERING total time
    """
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
        (patch_df["patch_time"].clip(lower=0) if include_patch_time else 0) +
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
               alpha=0.7, edgecolors='black', linewidth=0.2, color=color, s=120)
    
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
    
    stats_text = f"Avg: {mean_speedup:.2f}x\nMin: {min_speedup:.2f}x\nMax: {max_speedup:.2f}x"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def main():
    parser = argparse.ArgumentParser(description="Tet Mesh Speedup Analysis")
    parser.add_argument(
        "--include-patch-time",
        action="store_true",
        help="Include patch_time in PATCH_ORDERING total time calculation"
    )
    args = parser.parse_args()
    
    # Get the path to the data file
    script_dir = Path(__file__).parent
    data_file = script_dir / "data" / "tet_mesh_benchmark.csv"
    result_dir = script_dir / "results"
    
    # Read the CSV file
    df = pd.read_csv(data_file)
    
    print(f"Total entries: {len(df)}")
    print(f"Include patch_time: {args.include_patch_time}")
    
    # Compute speedup for both solvers
    cudss_df = compute_speedup_for_solver(df, "CUDSS", args.include_patch_time)
    mkl_df = compute_speedup_for_solver(df, "MKL", args.include_patch_time)
    
    # Create the plot with two subplots stacked vertically
    scale_factor = 2
    fig, (ax_cudss, ax_mkl) = plt.subplots(2, 1, figsize=(3.36 * scale_factor, 1.5 * scale_factor * 2))
    
    # Plot CUDSS on top
    plot_speedup(ax_cudss, cudss_df, "CUDSS")
    ax_cudss.set_xlabel(r"Number of mesh vertices")
    
    # Plot MKL on bottom (red color)
    plot_speedup(ax_mkl, mkl_df, "MKL", color='green')
    ax_mkl.set_xlabel(r"Number of mesh vertices")
    
    plt.tight_layout()
    
    # Save the figure with appropriate name
    suffix = "_with_patch" if args.include_patch_time else ""
    output_path = result_dir / f"tet_mesh_speedup{suffix}.pdf"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
