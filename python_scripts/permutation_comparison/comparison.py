#!/usr/bin/env python3
"""
Permutation Comparison Script

Computes and visualizes ORDERING speedup of PATCH_ORDERING over AMD, METIS, and ParMETIS orderings.

The ordering speedup is computed as:
- Baseline ordering time = baseline_analysis_time - patch_analysis_time
  (since baseline analysis includes internal ordering overhead, while patch analysis is "pure")
- Patch ordering time = patch_time + ordering_time
- Speedup = baseline_ordering_time / patch_ordering_time
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 12


def get_best_patch_ordering(df):
    """
    For each mesh, keep only the configuration with the smallest ordering runtime.
    Ordering runtime = patch_time + ordering_time + ordering_integration_time
    """
    df = df.copy()
    
    # Compute ordering runtime for selection
    df["patch_ordering_time"] = (
        df["patch_time"].clip(lower=0) +
        df["ordering_time"].clip(lower=0) +
        df["ordering_integration_time"].clip(lower=0)
    )
    
    # Select configuration with smallest ordering runtime
    best_df = df.loc[df.groupby("mesh_name")["patch_ordering_time"].idxmin()]
    return best_df


def compute_nnz_ratio_improvement(baseline_df, patch_df):
    """
    Compute NNZ ratio improvement of PATCH_ORDERING over a baseline ordering.
    
    Improvement = baseline_nnz_ratio / patch_nnz_ratio
    Values > 1 indicate patch ordering produces lower fill-in (better).
    """
    # Prepare baseline data
    baseline_for_merge = baseline_df[["mesh_name", "G_N", "factor/matrix NNZ ratio"]].rename(
        columns={"factor/matrix NNZ ratio": "baseline_nnz_ratio"}
    )
    
    # Prepare patch data (use best config which already has NNZ ratio)
    patch_for_merge = patch_df[["mesh_name", "factor/matrix NNZ ratio"]].rename(
        columns={"factor/matrix NNZ ratio": "patch_nnz_ratio"}
    )
    
    # Merge on mesh_name
    merged_df = pd.merge(baseline_for_merge, patch_for_merge, on="mesh_name", how="inner")
    
    # Compute improvement ratio (baseline / patch)
    # Values > 1 mean patch ordering is better (lower fill-in)
    merged_df["nnz_improvement"] = merged_df["baseline_nnz_ratio"] / merged_df["patch_nnz_ratio"]
    
    # Sort by G_N for cleaner plots
    merged_df = merged_df.sort_values("G_N")
    
    return merged_df


def compute_ordering_speedup(baseline_df, patch_df):
    """
    Compute ORDERING speedup of PATCH_ORDERING over a baseline ordering.
    
    Baseline ordering time = baseline_analysis_time - patch_analysis_time
    (The baseline's analysis_time includes internal ordering overhead,
     while patch ordering's analysis_time is "pure" analysis without ordering)
    
    Patch ordering time = patch_time + ordering_time + ordering_integration_time
    
    Speedup = baseline_ordering_time / patch_ordering_time
    
    For small meshes where the ordering time estimation fails (baseline_ordering_time <= 0),
    we fall back to comparing total analysis times:
    Speedup = baseline_analysis_time / (patch_analysis_time + patch_ordering_time)
    """
    # Prepare baseline data
    baseline_for_merge = baseline_df[["mesh_name", "G_N", "analysis_time"]].rename(
        columns={"analysis_time": "baseline_analysis_time"}
    )
    
    # Prepare patch data with ordering time and pure analysis time
    patch_for_merge = patch_df[["mesh_name", "analysis_time", "patch_ordering_time"]].rename(
        columns={"analysis_time": "patch_analysis_time"}
    )
    
    # Merge on mesh_name
    merged_df = pd.merge(baseline_for_merge, patch_for_merge, on="mesh_name", how="inner")
    
    # Compute baseline ordering time (embedded in analysis_time)
    # Subtract the "pure" analysis time (from patch ordering) to isolate ordering overhead
    merged_df["baseline_ordering_time"] = (
        merged_df["baseline_analysis_time"] - merged_df["patch_analysis_time"]
    )
    
    # Compute speedup: use ordering time comparison when valid,
    # otherwise fall back to total analysis time comparison
    merged_df["patch_total_time"] = merged_df["patch_analysis_time"] + merged_df["patch_ordering_time"]
    
    # Default: ordering time comparison
    merged_df["speedup"] = merged_df["baseline_ordering_time"] / merged_df["patch_ordering_time"]
    
    # Fallback for small meshes where ordering time estimation fails
    fallback_mask = merged_df["baseline_ordering_time"] <= 0
    merged_df.loc[fallback_mask, "speedup"] = (
        merged_df.loc[fallback_mask, "baseline_analysis_time"] / 
        merged_df.loc[fallback_mask, "patch_total_time"]
    )
    
    # Sort by G_N for cleaner plots
    merged_df = merged_df.sort_values("G_N")
    
    return merged_df


def plot_speedup(ax, merged_df, baseline_name, color='C0'):
    """Plot ordering speedup data on the given axes."""
    ax.scatter(merged_df["G_N"], merged_df["speedup"],
               alpha=0.7, edgecolors='black', linewidth=0.5, color=color)
    
    # Log scale for x-axis due to wide range of G_N values
    ax.set_xscale("log")
    
    # Horizontal line at speedup = 1 (no speedup)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    # Remove upper and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_ylabel(f"Ordering Speedup vs {baseline_name}")
    ax.grid(True, alpha=0.3)
    
    # Statistics text box
    mean_speedup = merged_df["speedup"].mean()
    min_speedup = merged_df["speedup"].min()
    max_speedup = merged_df["speedup"].max()
    
    stats_text = f"Mean: {mean_speedup:.2f}x\nMin: {min_speedup:.2f}x\nMax: {max_speedup:.2f}x"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_nnz_ratio(ax, merged_df, baseline_name, color='C0'):
    """Plot NNZ ratio improvement data on the given axes."""
    ax.scatter(merged_df["G_N"], merged_df["nnz_improvement"],
               alpha=0.7, edgecolors='black', linewidth=0.5, color=color)
    
    # Log scale for x-axis due to wide range of G_N values
    ax.set_xscale("log")
    
    # Horizontal line at ratio = 1 (no improvement)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    # Remove upper and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_ylabel(f"Patch Fill-ratio vs {baseline_name}")
    ax.grid(True, alpha=0.3)
    
    # Statistics text box
    mean_ratio = merged_df["nnz_improvement"].mean()
    min_ratio = merged_df["nnz_improvement"].min()
    max_ratio = merged_df["nnz_improvement"].max()
    
    stats_text = f"Mean: {mean_ratio:.2f}x\nMin: {min_ratio:.2f}x\nMax: {max_ratio:.2f}x"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def main():
    # Paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    result_dir = script_dir / "results"
    result_dir.mkdir(exist_ok=True)
    
    # Load data
    amd_df = pd.read_csv(data_dir / "amd_benchmark.csv")
    metis_df = pd.read_csv(data_dir / "metis_benchmark.csv")
    parmetis_df = pd.read_csv(data_dir / "parmetis_benchmark.csv")
    patch_df = pd.read_csv(data_dir / "patch_amd_benchmark.csv")
    
    print(f"AMD entries: {len(amd_df)}")
    print(f"METIS entries: {len(metis_df)}")
    print(f"ParMETIS entries: {len(parmetis_df)}")
    print(f"Patch ordering entries: {len(patch_df)}")
    
    # Get best configuration for patch ordering
    patch_best_df = get_best_patch_ordering(patch_df)
    print(f"Patch ordering (best config) entries: {len(patch_best_df)}")
    
    # Compute ordering speedups
    amd_speedup = compute_ordering_speedup(amd_df, patch_best_df)
    metis_speedup = compute_ordering_speedup(metis_df, patch_best_df)
    parmetis_speedup = compute_ordering_speedup(parmetis_df, patch_best_df)
    
    print(f"Matched pairs (AMD): {len(amd_speedup)}")
    print(f"Matched pairs (METIS): {len(metis_speedup)}")
    print(f"Matched pairs (ParMETIS): {len(parmetis_speedup)}")
    
    # Compute NNZ ratio improvements
    amd_nnz = compute_nnz_ratio_improvement(amd_df, patch_best_df)
    metis_nnz = compute_nnz_ratio_improvement(metis_df, patch_best_df)
    parmetis_nnz = compute_nnz_ratio_improvement(parmetis_df, patch_best_df)
    
    print(f"NNZ ratio pairs (AMD): {len(amd_nnz)}")
    print(f"NNZ ratio pairs (METIS): {len(metis_nnz)}")
    print(f"NNZ ratio pairs (ParMETIS): {len(parmetis_nnz)}")
    
    # Create figure with 3 subplots for ordering speedup
    scale_factor = 2
    fig1, (ax_amd, ax_metis, ax_parmetis) = plt.subplots(
        3, 1, figsize=(3.36 * scale_factor, 1.5 * scale_factor * 3)
    )
    
    # Plot speedups
    plot_speedup(ax_amd, amd_speedup, "AMD", color='C0')
    plot_speedup(ax_metis, metis_speedup, "METIS", color='C1')
    plot_speedup(ax_parmetis, parmetis_speedup, "ParMETIS", color='C2')
    
    # Only set x-label on bottom plot
    ax_parmetis.set_xlabel("Number of mesh vertices")
    
    plt.tight_layout()
    
    # Save the speedup figure
    output_path = result_dir / "permutation_comparison.pdf"
    plt.savefig(output_path, dpi=150)
    print(f"Speedup plot saved to: {output_path}")
    
    # Create figure with 3 subplots for NNZ ratio comparison
    fig2, (ax_amd_nnz, ax_metis_nnz, ax_parmetis_nnz) = plt.subplots(
        3, 1, figsize=(3.36 * scale_factor, 1.5 * scale_factor * 3)
    )
    
    # Plot NNZ ratio improvements
    plot_nnz_ratio(ax_amd_nnz, amd_nnz, "AMD", color='C0')
    plot_nnz_ratio(ax_metis_nnz, metis_nnz, "METIS", color='C1')
    plot_nnz_ratio(ax_parmetis_nnz, parmetis_nnz, "ParMETIS", color='C2')
    
    # Only set x-label on bottom plot
    ax_parmetis_nnz.set_xlabel("Number of mesh vertices")
    
    plt.tight_layout()
    
    # Save the NNZ ratio figure
    nnz_output_path = result_dir / "nnz_ratio_comparison.pdf"
    plt.savefig(nnz_output_path, dpi=150)
    print(f"NNZ ratio plot saved to: {nnz_output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()

