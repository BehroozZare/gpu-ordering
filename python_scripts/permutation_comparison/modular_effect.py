#!/usr/bin/env python3
"""
Modular Effect Script

Visualizes the effect of local permutation method (AMD vs METIS)
on fill-in ratio and runtime for PATCH_ORDERING on two meshes 
(Large: Murex_Romosus, Small: Aloisus_C).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib as mpl

# Matplotlib configuration for publication-quality figures
# mpl.rcParams['pdf.fonttype'] = 42
# mpl.rcParams['ps.fonttype'] = 42
# mpl.rcParams['font.size'] = 12

mpl.rcParams['font.family'] = ['Palatino Linotype', 'serif']
mpl.rcParams['font.size'] = 12


def load_data():
    """Load and filter local permutation comparison data from CSV."""
    script_dir = Path(__file__).parent
    data_path = script_dir / "data" / "module_select.csv"
    df = pd.read_csv(data_path)
    
    # Filter: patch_type=rxmesh, use_patch_separator=1 (Patch-based separator)
    df = df[(df["patch_type"] == "rxmesh") & (df["use_patch_separator"] == 1)]
    
    # Filter to only include the meshes we want
    df = df[df["mesh_name"].isin(["Aloisus_C", "Murex_Romosus"])]
    
    # Create display names
    df = df.copy()
    df["size_label"] = df["mesh_name"].map({
        "Murex_Romosus": "Large",
        "Aloisus_C": "Small"
    })
    
    df["method_label"] = df["local_permute_method"].str.upper()
    
    # Create combined label
    df["display_name"] = df["size_label"] + "-" + df["method_label"]
    
    return df


def compute_runtime(df):
    """
    Compute runtime for local permutation comparison.
    
    Uses local_permute_time (in seconds) converted to ms.
    """
    df = df.copy()
    
    # Convert seconds to ms for local_permute_time
    df["computed_runtime_ms"] = df["local_permute_time"] * 1000
    
    return df


def get_y_positions_with_gap(n_per_group=2, gap=0.5):
    """
    Create y positions with a gap between Large and Small groups.
    Returns y positions and the gap size for consistent use.
    
    In horizontal bar charts, lower y = bottom, higher y = top.
    We want Small at top (first when reading), Large at bottom.
    """
    # First group (Large - bottom): positions 0, 1
    # Gap
    # Second group (Small - top): positions 2+gap, 3+gap
    y_first_group = np.arange(n_per_group)
    y_second_group = np.arange(n_per_group) + n_per_group + gap
    return np.concatenate([y_first_group, y_second_group]), gap


def plot_fill_in_ratio(ax, df):
    """
    Create horizontal bar chart for fill-in ratio across local permutation methods.
    """
    # Sort data: Large first (bottom), then Small (top - first when reading)
    # Within each group: AMD first, then METIS
    df_sorted = df.sort_values(
        by=["size_label", "local_permute_method"],
        ascending=[True, True]  # Large before Small alphabetically (so Small at top), AMD before METIS
    )
    
    labels = df_sorted["display_name"].values
    fill_in_values = df_sorted["factor/matrix NNZ ratio"].values
    
    # Get y positions with gap between groups
    y, _ = get_y_positions_with_gap(n_per_group=2, gap=0.5)
    
    # Use different colors for different local permutation methods
    colors = ['coral' if 'METIS' in label else 'plum' for label in labels]
    
    ax.barh(y, fill_in_values, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y)
    ax.set_yticklabels(labels, ha='left')
    ax.set_xlabel("Factor / Matrix NNZ")
    
    # Adjust y-axis tick label position to left align
    ax.tick_params(axis='y', pad=80)
    
    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.grid(True, alpha=0.3, axis='x')


def plot_runtime(ax, df):
    """
    Create horizontal bar chart for runtime comparison.
    
    Runtime is based on local_permute_time.
    """
    # Compute runtime
    df = compute_runtime(df)
    
    # Sort data: Large first (bottom), then Small (top - first when reading)
    # Within each group: AMD first, then METIS
    df_sorted = df.sort_values(
        by=["size_label", "local_permute_method"],
        ascending=[True, True]  # Large before Small alphabetically (so Small at top), AMD before METIS
    )
    
    labels = df_sorted["display_name"].values
    runtime_values = df_sorted["computed_runtime_ms"].values
    
    # Get y positions with gap between groups
    y, _ = get_y_positions_with_gap(n_per_group=2, gap=0.5)
    
    # Use different colors for different local permutation methods
    colors = ['coral' if 'METIS' in label else 'plum' for label in labels]
    
    ax.barh(y, runtime_values, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y)
    ax.set_yticklabels(labels, ha='left')
    ax.set_xlabel("Local Permutation Time (ms)")
    
    # Adjust y-axis tick label position to left align
    ax.tick_params(axis='y', pad=80)
    
    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='plum', edgecolor='black', label='AMD'),
        Patch(facecolor='coral', edgecolor='black', label='METIS')
    ]
    ax.legend(handles=legend_elements, loc='upper right')


def print_comparison_stats(df):
    """Print fill-in and runtime comparison statistics between AMD and METIS."""
    df = compute_runtime(df)
    
    print("\n" + "=" * 60)
    print("LOCAL PERMUTATION METHOD COMPARISON STATISTICS")
    print("=" * 60)
    
    for size_label in ["Small", "Large"]:
        mesh_df = df[df["size_label"] == size_label]
        
        amd_row = mesh_df[mesh_df["local_permute_method"] == "amd"].iloc[0]
        metis_row = mesh_df[mesh_df["local_permute_method"] == "metis"].iloc[0]
        
        # Factor / Matrix NNZ
        amd_factor_nnz = amd_row["factor/matrix NNZ ratio"]
        metis_factor_nnz = metis_row["factor/matrix NNZ ratio"]
        factor_nnz_ratio = amd_factor_nnz / metis_factor_nnz
        
        # Runtime
        amd_runtime = amd_row["computed_runtime_ms"]
        metis_runtime = metis_row["computed_runtime_ms"]
        runtime_ratio = amd_runtime / metis_runtime
        
        print(f"\n{size_label} mesh ({amd_row['mesh_name']}):")
        print(f"  Factor / Matrix NNZ:")
        print(f"    AMD:   {amd_factor_nnz:.4f}")
        print(f"    METIS: {metis_factor_nnz:.4f}")
        print(f"    Ratio (AMD/METIS): {factor_nnz_ratio:.4f} ({(factor_nnz_ratio - 1) * 100:+.2f}%)")
        print(f"  Local Permutation Time (ms):")
        print(f"    AMD:   {amd_runtime:.2f} ms")
        print(f"    METIS: {metis_runtime:.2f} ms")
        print(f"    Ratio (AMD/METIS): {runtime_ratio:.4f} ({(runtime_ratio - 1) * 100:+.2f}%)")
        print(f"    Speedup (METIS/AMD): {1/runtime_ratio:.2f}x")
    
    print("\n" + "=" * 60 + "\n")


def main():
    # Paths
    script_dir = Path(__file__).parent
    result_dir = script_dir / "results"
    result_dir.mkdir(exist_ok=True)
    
    # Load data
    df = load_data()
    print(f"Loaded {len(df)} entries")
    print(f"Meshes: {df['mesh_name'].unique()}")
    print(f"Local permutation methods: {df['local_permute_method'].unique()}")
    print(f"Display names: {df['display_name'].values}")
    
    # Print comparison statistics
    print_comparison_stats(df)
    
    # Create figure with 2 subplots (both horizontal bar charts)
    # Runtime on top, fill-in on bottom
    scale_factor = 2
    fig, (ax_runtime, ax_fill_in) = plt.subplots(
        2, 1, figsize=(3.36 * scale_factor, 1.2 * scale_factor * 2),
        gridspec_kw={'height_ratios': [1, 1]}
    )
    
    # Plot 1: Runtime comparison (top)
    plot_runtime(ax_runtime, df)
    
    # Plot 2: Fill-in ratio bar chart (bottom)
    plot_fill_in_ratio(ax_fill_in, df)
    
    plt.tight_layout()
    
    # Save figure
    output_path = result_dir / "modular_effect.pdf"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
