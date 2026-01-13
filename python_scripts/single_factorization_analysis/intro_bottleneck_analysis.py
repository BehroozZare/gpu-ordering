#!/usr/bin/env python3
"""
Bottleneck Analysis Script

Analyzes the ratio of symbolic analysis time to total linear solver time
for CUDSS solver with DEFAULT ordering.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

# mpl.rcParams['pdf.fonttype'] = 42
# mpl.rcParams['ps.fonttype'] = 42
# mpl.rcParams['font.size'] = 12

mpl.rcParams['font.family'] = ['Palatino Linotype', 'serif']
mpl.rcParams['font.size'] = 18

def main():
    # Get the path to the data file
    script_dir = Path(__file__).parent
    data_file = script_dir / "data" / "laplace_full_benchmark.csv"
    result_dir = script_dir / "results"
    
    # Read the CSV file
    df = pd.read_csv(data_file)
    
    # Filter for DEFAULT ordering_type and CUDSS solver_type
    filtered_df = df[(df["ordering_type"] == "DEFAULT") & (df["solver_type"] == "CUDSS")].copy()
    
    print(f"Total entries: {len(df)}")
    print(f"Filtered entries (DEFAULT + CUDSS): {len(filtered_df)}")
    
    # Compute total linear solver time
    # Total = ordering_integration_time + analysis_time + factorization_time + solve_time
    filtered_df["total_solver_time"] = (
        filtered_df["ordering_integration_time"] +
        filtered_df["analysis_time"] +
        filtered_df["factorization_time"] +
        filtered_df["solve_time"]
    )
    
    # Compute symbolic analysis time
    # Symbolic = ordering_integration_time + analysis_time
    filtered_df["symbolic_analysis_time"] = (
        filtered_df["ordering_integration_time"]
        # filtered_df["analysis_time"]
    )
    
    # Compute percentage of symbolic analysis time to total solver time
    filtered_df["symbolic_percentage"] = (
        filtered_df["symbolic_analysis_time"] / filtered_df["total_solver_time"]
    ) * 100
    
    # Sort by G_N for a cleaner plot
    filtered_df = filtered_df.sort_values("G_N")
    
    # Create the plot
    scale_factor = 2
    fig, ax = plt.subplots(figsize=(3.36 * scale_factor, 2 * scale_factor))
    
    ax.scatter(filtered_df["G_N"], filtered_df["symbolic_percentage"], 
               alpha=0.8, edgecolors='black', linewidth=0.2, s=120)
    
    # Use log scale for x-axis due to wide range of G_N values
    ax.set_xscale("log")
    
    # Remove upper and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlabel(r"#V")
    ax.set_ylabel("Permutation time / Total time (%)")
    #ax.set_title("Ordering Bottleneck (CUDSS)")
    
    ax.grid(True, alpha=0.3)
    
    # Add some statistics as text
    mean_pct = filtered_df["symbolic_percentage"].mean()
    min_pct = filtered_df["symbolic_percentage"].min()
    max_pct = filtered_df["symbolic_percentage"].max()
    
    stats_text = f"Avg:  {mean_pct:.1f}%\nMin:  {min_pct:.1f}%\nMax:  {max_pct:.1f}%"
    ax.text(0.5, 0.5, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save the figure
    output_path = result_dir / "intro_ordering_bottleneck.pdf"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()

