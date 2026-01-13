#!/usr/bin/env python3
"""
Runtime Breakdown Plot

Creates a horizontal stacked bar plot showing the runtime breakdown
of the GPU ordering algorithm for two meshes (large and small).
Each segment represents a normalized percentage of the total runtime.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
import numpy as np

# PDF font settings for publication-quality figures
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 12


def main():
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    data_path = project_root / "output" / "patch_ordering_ablation" / "patch_size_effect.csv"
    result_dir = script_dir / "results"
    result_dir.mkdir(exist_ok=True)
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Filter for specific configuration
    filtered_df = df[
        (df["use_patch_separator"] == 1) &
        (df["local_permute_method"] == "amd") &
        (df["patch_size"] == 512)
    ].copy()
    
    # Map mesh names to display labels
    mesh_name_map = {
        "Murex_Romosus": "Large",
        "dragon": "Small"
    }
    filtered_df["display_name"] = filtered_df["mesh_name"].map(mesh_name_map)
    
    # Sort so Large is on top (reverse order for barh)
    filtered_df = filtered_df.sort_values("G_N", ascending=True)
    
    # Define the time components and their labels
    # patch_time is in ms, others are in seconds - convert to ms
    time_columns = [
        ("patch_time", "Patching"),
        ("node_to_patch_time", "Quotient"),
        ("decompose_time", "Etree"),
        ("local_permute_time", "Local Permute"),
        ("assemble_time", "Concatenation"),
    ]
    
    # Convert seconds to milliseconds for consistency (patch_time is already in ms)
    for col, _ in time_columns[1:]:  # Skip patch_time
        filtered_df[col] = filtered_df[col] * 1000
    
    # Compute total time and normalized percentages
    total_times = filtered_df[[col for col, _ in time_columns]].sum(axis=1)
    
    # Prepare data for plotting
    mesh_labels = filtered_df["display_name"].tolist()
    
    # Create figure
    scale_factor = 2
    fig, ax = plt.subplots(figsize=(3.36 * scale_factor, 1.0 * scale_factor))
    
    # Colors for each segment
    colors = plt.cm.tab10(np.linspace(0, 1, len(time_columns)))
    
    # Create stacked horizontal bars
    left_positions = np.zeros(len(filtered_df))
    
    for i, (col, label) in enumerate(time_columns):
        # Normalize to percentage (0 to 1)
        values = (filtered_df[col].values / total_times.values)
        
        ax.barh(mesh_labels, values, left=left_positions, 
                label=label, color=colors[i], edgecolor='white', linewidth=0.5)
        
        left_positions += values
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlim(0, 1)
    
    # Legend on top of the plot (two rows)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
              ncol=3, frameon=False, fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = result_dir / "runtime_breakdown.pdf"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Runtime breakdown plot saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
