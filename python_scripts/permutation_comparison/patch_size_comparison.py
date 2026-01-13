#!/usr/bin/env python3
"""
Patch Size Comparison Script

Visualizes the effect of patch size on fill-in ratio and runtime breakdown
for PATCH_ORDERING on two meshes (Large: Murex_Romosus, Small: dragon).
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
    """Load patch size effect data from CSV."""
    script_dir = Path(__file__).parent
    # Data is in the output directory
    data_path = script_dir.parent.parent / "output" / "patch_ordering_ablation" / "patch_size_effect.csv"
    df = pd.read_csv(data_path)
    
    # Create display names for meshes
    df["display_name"] = df["mesh_name"].map({
        "Murex_Romosus": "Large",
        "Aloisus_C": "Small"
    })
    
    # Filter to only include the meshes we want
    df = df[df["display_name"].notna()]
    
    # Remove patch size 64
    df = df[df["patch_size"] != 64]
    
    return df


def get_y_positions_with_gap(n_per_group=4, gap=0.5):
    """
    Create y positions with a gap between Large and Small groups.
    Returns y positions and the gap size for consistent use.
    """
    # First group (Small): positions 0, 1, 2, 3
    # Gap
    # Second group (Large): positions 4+gap, 5+gap, 6+gap, 7+gap
    y_small = np.arange(n_per_group)
    y_large = np.arange(n_per_group) + n_per_group + gap
    return np.concatenate([y_small, y_large]), gap


def plot_fill_in_ratio(ax, df):
    """
    Create horizontal bar chart for fill-in ratio across patch sizes.
    
    Same layout as runtime breakdown: 8 horizontal bars
    """
    # Sort data: Small first (bottom), then Large (top), each sorted by patch_size
    df_sorted = df.sort_values(
        by=["display_name", "patch_size"],
        ascending=[True, True]  # Small before Large, patch_size ascending
    )
    
    # Create labels for y-axis
    labels = [f"{row['display_name']}-{row['patch_size']}" 
              for _, row in df_sorted.iterrows()]
    
    fill_in_values = df_sorted["factor/matrix NNZ ratio"].values
    
    # Get y positions with gap between groups
    y, _ = get_y_positions_with_gap(n_per_group=3, gap=0.5)
    
    ax.barh(y, fill_in_values, color='C0', edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Factor / Matrix NNZ")
    
    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.grid(True, alpha=0.3, axis='x')


def plot_runtime_breakdown(ax, df):
    """
    Create horizontal stacked bar chart for normalized runtime breakdown.
    
    8 bars: one for each (mesh, patch_size) combination
    Each bar normalized to 1 (100%)
    
    Note: patch_time is in ms, other times are in seconds.
    Convert all to ms for consistency before normalizing.
    """
    # Runtime components and their display labels
    # (column_name, display_label, is_in_seconds)
    components = [
        ("patch_time", "Patch", False),           # already in ms
        ("node_to_patch_time", "Quot.", True),    # in seconds
        ("decompose_time", "etree", True),        # in seconds
        ("local_permute_time", "Perm.", True),    # in seconds
        ("assemble_time", "Concat.", True),       # in seconds
    ]
    
    # Sort data: Small first (bottom), then Large (top), each sorted by patch_size
    df_sorted = df.sort_values(
        by=["display_name", "patch_size"],
        ascending=[True, True]  # Small before Large, patch_size ascending
    ).copy()
    
    # Convert seconds to ms for relevant columns
    for col, label, is_seconds in components:
        if is_seconds:
            df_sorted[col] = df_sorted[col] * 1000  # seconds to ms
    
    # Create labels for y-axis
    labels = [f"{row['display_name']}-{row['patch_size']}" 
              for _, row in df_sorted.iterrows()]
    
    # Compute total runtime for each row
    component_cols = [c[0] for c in components]
    df_sorted["total_runtime"] = df_sorted[component_cols].sum(axis=1)
    
    # Get normalization factors: total runtime of patch_size=512 for each mesh
    norm_factors = {}
    for mesh in ["Small", "Large"]:
        mesh_512 = df_sorted[(df_sorted["display_name"] == mesh) & 
                             (df_sorted["patch_size"] == 512)]
        norm_factors[mesh] = mesh_512["total_runtime"].values[0]
    
    # Create normalization array matching df_sorted order
    norm_array = df_sorted["display_name"].map(norm_factors).values
    
    # Prepare normalized data (normalized to respective 512 configuration)
    normalized_data = {}
    for col, label, _ in components:
        normalized_data[label] = (df_sorted[col] / norm_array).values
    
    # Create stacked horizontal bars with gap between groups
    y, _ = get_y_positions_with_gap(n_per_group=3, gap=0.5)
    left = np.zeros(len(labels))
    
    colors = ['C9', 'C1', 'C2', 'C3', 'C4']  # C9=cyan, avoids C0 (blue) used in fill-in plot
    
    for i, (col, label, _) in enumerate(components):
        values = normalized_data[label]
        ax.barh(y, values, left=left, label=label, color=colors[i],
                edgecolor='black', linewidth=0.5)
        left += values
    
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Normalized Runtime (relative to patch size 512)")
    
    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend at the top of the plot (all 5 labels in one row)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=5, fontsize=10)


def main():
    # Paths
    script_dir = Path(__file__).parent
    result_dir = script_dir / "results"
    result_dir.mkdir(exist_ok=True)
    
    # Load data
    df = load_data()
    print(f"Loaded {len(df)} entries")
    print(f"Meshes: {df['mesh_name'].unique()}")
    print(f"Patch sizes: {sorted(df['patch_size'].unique())}")
    
    # Create figure with 2 subplots (both horizontal bar charts)
    # Runtime on top, fill-in on bottom
    scale_factor = 2
    fig, (ax_runtime, ax_fill_in) = plt.subplots(
        2, 1, figsize=(3.36 * scale_factor, 1.5 * scale_factor * 2),
        gridspec_kw={'height_ratios': [1, 1]}
    )
    
    # Plot 1: Runtime breakdown stacked bars (top)
    plot_runtime_breakdown(ax_runtime, df)
    
    # Plot 2: Fill-in ratio bar chart (bottom)
    plot_fill_in_ratio(ax_fill_in, df)
    
    plt.tight_layout()
    
    # Save figure
    output_path = result_dir / "patch_size_comparison.pdf"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
