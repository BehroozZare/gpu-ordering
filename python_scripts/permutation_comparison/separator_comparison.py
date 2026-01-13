#!/usr/bin/env python3
"""
Separator Comparison Script

Visualizes the effect of separator computation method (patch-based vs METIS)
on fill-in ratio and runtime for PATCH_ORDERING on two meshes 
(Large: Murex_Romosus, Small: Aloisus_C).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib as mpl

# Matplotlib configuration for publication-quality figures
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 12


def load_data():
    """Load and filter separator comparison data from CSV."""
    script_dir = Path(__file__).parent
    data_path = script_dir / "data" / "module_select.csv"
    df = pd.read_csv(data_path)
    
    # Filter: local_permute_method=amd, patch_type=rxmesh
    df = df[(df["local_permute_method"] == "amd") & (df["patch_type"] == "rxmesh")]
    
    # Filter to only include the meshes we want
    df = df[df["mesh_name"].isin(["Aloisus_C", "Murex_Romosus"])]
    
    # Create display names
    df = df.copy()
    df["size_label"] = df["mesh_name"].map({
        "Murex_Romosus": "Large",
        "Aloisus_C": "Small"
    })
    
    df["separator_label"] = df["use_patch_separator"].map({
        1: "Patch",
        0: "METIS"
    })
    
    # Create combined label
    df["display_name"] = df["size_label"] + "-" + df["separator_label"]
    
    return df


def compute_runtime(df):
    """
    Compute runtime based on separator type.
    
    - METIS separator (use_patch_separator=0): decompose_time only
    - Patch-based separator (use_patch_separator=1): patch_time + node_to_patch_time + decompose_time
    
    Note: patch_time is in ms, other times are in seconds. Convert all to ms.
    """
    df = df.copy()
    
    # Convert seconds to ms for time columns
    df["decompose_time_ms"] = df["decompose_time"] * 1000
    df["node_to_patch_time_ms"] = df["node_to_patch_time"] * 1000
    # patch_time is already in ms
    
    # Compute runtime based on separator type
    runtime = np.where(
        df["use_patch_separator"] == 0,
        df["decompose_time_ms"],  # METIS separator: only decompose_time
        df["patch_time"] + df["node_to_patch_time_ms"] + df["decompose_time_ms"]  # Patch-based
    )
    
    df["computed_runtime_ms"] = runtime
    
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
    Create horizontal bar chart for fill-in ratio across separator types.
    """
    # Sort data: Large first (bottom), then Small (top - first when reading)
    # Within each group: METIS first, then Patch
    df_sorted = df.sort_values(
        by=["size_label", "use_patch_separator"],
        ascending=[True, True]  # Large before Small alphabetically (so Small at top), METIS (0) before Patch (1)
    )
    
    labels = df_sorted["display_name"].values
    fill_in_values = df_sorted["factor/matrix NNZ ratio"].values
    
    # Get y positions with gap between groups
    y, _ = get_y_positions_with_gap(n_per_group=2, gap=0.5)
    
    # Use different colors for different separator types
    colors = ['C1' if 'METIS' in label else 'C0' for label in labels]
    
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
    
    Runtime calculation:
    - METIS separator: decompose_time only
    - Patch-based separator: patch_time + node_to_patch_time + decompose_time
    """
    # Compute runtime
    df = compute_runtime(df)
    
    # Sort data: Large first (bottom), then Small (top - first when reading)
    # Within each group: METIS first, then Patch
    df_sorted = df.sort_values(
        by=["size_label", "use_patch_separator"],
        ascending=[True, True]  # Large before Small alphabetically (so Small at top), METIS (0) before Patch (1)
    )
    
    labels = df_sorted["display_name"].values
    runtime_values = df_sorted["computed_runtime_ms"].values
    
    # Get y positions with gap between groups
    y, _ = get_y_positions_with_gap(n_per_group=2, gap=0.5)
    
    # Use different colors for different separator types
    colors = ['C1' if 'METIS' in label else 'C0' for label in labels]
    
    ax.barh(y, runtime_values, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y)
    ax.set_yticklabels(labels, ha='left')
    ax.set_xlabel("Runtime (ms)")
    
    # Adjust y-axis tick label position to left align
    ax.tick_params(axis='y', pad=80)
    
    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='C0', edgecolor='black', label='Patch-based separator'),
        Patch(facecolor='C1', edgecolor='black', label='METIS separator')
    ]
    ax.legend(handles=legend_elements, loc='upper right')


def print_comparison_stats(df):
    """Print fill-in and runtime comparison statistics between Patch and METIS separators."""
    df = compute_runtime(df)
    
    print("\n" + "=" * 60)
    print("SEPARATOR COMPARISON STATISTICS")
    print("=" * 60)
    
    for size_label in ["Small", "Large"]:
        mesh_df = df[df["size_label"] == size_label]
        
        patch_row = mesh_df[mesh_df["use_patch_separator"] == 1].iloc[0]
        metis_row = mesh_df[mesh_df["use_patch_separator"] == 0].iloc[0]
        
        # Factor / Matrix NNZ
        patch_factor_nnz = patch_row["factor/matrix NNZ ratio"]
        metis_factor_nnz = metis_row["factor/matrix NNZ ratio"]
        factor_nnz_ratio = patch_factor_nnz / metis_factor_nnz
        
        # Runtime
        patch_runtime = patch_row["computed_runtime_ms"]
        metis_runtime = metis_row["computed_runtime_ms"]
        runtime_ratio = patch_runtime / metis_runtime
        
        print(f"\n{size_label} mesh ({patch_row['mesh_name']}):")
        print(f"  Factor / Matrix NNZ:")
        print(f"    Patch-based: {patch_factor_nnz:.4f}")
        print(f"    METIS:       {metis_factor_nnz:.4f}")
        print(f"    Ratio (Patch/METIS): {factor_nnz_ratio:.4f} ({(factor_nnz_ratio - 1) * 100:+.2f}%)")
        print(f"  Runtime (ms):")
        print(f"    Patch-based: {patch_runtime:.2f} ms")
        print(f"    METIS:       {metis_runtime:.2f} ms")
        print(f"    Ratio (Patch/METIS): {runtime_ratio:.4f} ({(runtime_ratio - 1) * 100:+.2f}%)")
        print(f"    Speedup (METIS/Patch): {1/runtime_ratio:.2f}x")
    
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
    print(f"Separator types: {df['use_patch_separator'].unique()}")
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
    output_path = result_dir / "separator_comparison.pdf"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
