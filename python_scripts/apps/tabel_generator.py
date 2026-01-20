#!/usr/bin/env python3
"""
Table Generator for Application Benchmark

Generates data for the LaTeX table showing application speedups and break-even
iteration counts for PATCH_ORDERING vs DEFAULT ordering.
"""

import pandas as pd
from pathlib import Path
import math

# Application settings mapping
APP_SETTINGS = {
    "SCP": "S1,F1,R2",
    "Smoothing": "S1,F2,R3",
    "NoiseSmoothing": "S1,F1,R1",
    "IPC": "S2,F2,R1",
    "IPC_no_col": "S1,F2,R1",
}


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
    Formula: patch_time + ordering_time + ordering_integration_time + analysis_time + factorization_time + solve_time
    Note: ordering_init_time is excluded per specification.
    Note: patch_time is stored as a constant in every row but should only be counted at iteration 0.
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


def process_mesh_data(df, mesh_name):
    """
    Process data for a single mesh and compute speedup and max_iter.
    
    Args:
        df: DataFrame with benchmark data
        mesh_name: Name of the mesh to process
        
    Returns:
        dict with mesh_name, G_N, num_iter, speedup, max_iter
    """
    # Filter for this mesh
    mesh_df = df[df["mesh_name"] == mesh_name].copy()
    
    # Separate DEFAULT and PATCH_ORDERING
    default_df = mesh_df[mesh_df["ordering_type"] == "DEFAULT"].copy()
    patch_df = mesh_df[mesh_df["ordering_type"] == "PATCH_ORDERING"].copy()
    
    if default_df.empty or patch_df.empty:
        print(f"  Warning: Missing data for mesh {mesh_name}")
        return None
    
    # Compute total_time for each row
    default_df["total_time"] = default_df.apply(compute_total_time_default, axis=1)
    patch_df["total_time"] = patch_df.apply(compute_total_time_patch, axis=1)
    
    # Speedup: sum all iterations
    default_total = default_df["total_time"].sum()
    patch_total = patch_df["total_time"].sum()
    speedup = default_total / patch_total if patch_total > 0 else float('inf')
    
    # Max iter calculation
    # Init time = iteration 0
    default_init = default_df[default_df["iteration"] == 0]["total_time"].values
    patch_init = patch_df[patch_df["iteration"] == 0]["total_time"].values
    
    if len(default_init) == 0 or len(patch_init) == 0:
        print(f"  Warning: No iteration 0 found for mesh {mesh_name}")
        return None
    
    default_init = default_init[0]
    patch_init = patch_init[0]
    
    # Avg iter time = mean of iterations > 0
    default_iter_df = default_df[default_df["iteration"] > 0]
    patch_iter_df = patch_df[patch_df["iteration"] > 0]
    
    if default_iter_df.empty or patch_iter_df.empty:
        # Only 1 iteration, max_iter is not applicable
        avg_default = 0
        avg_patch = 0
    else:
        avg_default = default_iter_df["total_time"].mean()
        avg_patch = patch_iter_df["total_time"].mean()
    
    # Solve: init_patch + (n-1)*avg_patch > init_default + (n-1)*avg_default
    # n = (init_patch - init_default) / (avg_default - avg_patch) + 1
    
    if avg_default >= avg_patch:
        # PATCH is always faster or equal per iteration
        max_iter = float('inf')
    else:
        denominator = avg_default - avg_patch
        if abs(denominator) < 1e-9:
            max_iter = float('inf')
        else:
            max_iter = (patch_init - default_init) / denominator + 1
            max_iter = max(1, int(max_iter))  # At least 1 iteration
    
    # Get mesh size (G_N) and number of iterations
    G_N = mesh_df["G_N"].iloc[0]
    num_iter = len(default_df)
    
    return {
        "mesh_name": mesh_name,
        "G_N": G_N,
        "num_iter": num_iter,
        "speedup": speedup,
        "max_iter": max_iter,
    }


def process_app_data(app_name, csv_path):
    """
    Process all meshes for a given application.
    
    Args:
        app_name: Name of the application
        csv_path: Path to the CSV file
        
    Returns:
        List of result dicts for each mesh
    """
    print(f"\nProcessing {app_name} from {csv_path}")
    
    if not csv_path.exists():
        print(f"  Warning: CSV file not found: {csv_path}")
        return []
    
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} entries")
    
    # Get unique mesh names
    mesh_names = df["mesh_name"].unique()
    print(f"  Found {len(mesh_names)} meshes: {list(mesh_names)}")
    
    results = []
    for mesh_name in mesh_names:
        result = process_mesh_data(df, mesh_name)
        if result:
            result["app_name"] = app_name
            result["setting"] = APP_SETTINGS.get(app_name, "")
            results.append(result)
    
    return results


def format_speedup(speedup):
    """Format speedup value for display."""
    if math.isinf(speedup):
        return "inf"
    return f"{speedup:.2f}x"


def format_max_iter(max_iter):
    """Format max_iter value for display."""
    if math.isinf(max_iter):
        return "inf"
    return str(int(max_iter))


def generate_latex_table_rows(results):
    """Generate LaTeX table rows from results."""
    lines = []
    current_app = None
    for r in results:
        # Add horizontal line between different applications
        if current_app is not None and current_app != r['app_name']:
            lines.append("    \\hline")
        current_app = r['app_name']
        
        # Escape underscores in mesh name for LaTeX
        mesh_name_latex = r['mesh_name'].replace("_", "\\_")
        line = f"    {r['app_name']:13} & {r['setting']:8} & {mesh_name_latex:25} & {r['G_N']:>10} & {r['num_iter']:5} & {format_speedup(r['speedup']):>10} & {format_max_iter(r['max_iter']):>8} \\\\"
        lines.append(line)
    return "\n".join(lines)


def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    print("=" * 60)
    print("Application Benchmark Table Generator")
    print("=" * 60)
    
    # Store results grouped by application
    results_by_app = {}
    
    # Process each application
    for app_name in APP_SETTINGS.keys():
        # Convert app name to folder/file name (lowercase, handle special cases)
        folder_name = app_name
        file_name = app_name.lower().replace("_", "_") + ".csv"
        
        csv_path = data_dir / folder_name / file_name
        
        results = process_app_data(app_name, csv_path)
        if results:
            results_by_app[app_name] = results
    
    # Print results grouped by application
    print("\n" + "=" * 80)
    print("Results Summary (By Application)")
    print("=" * 80)
    
    if not results_by_app:
        print("No results to display. Make sure CSV files exist in the data folder.")
        return
    
    all_results = []
    
    for app_name, app_results in results_by_app.items():
        setting = APP_SETTINGS.get(app_name, "")
        print(f"\n{'='*80}")
        print(f"Application: {app_name} (Setting: {setting})")
        print(f"{'='*80}")
        print(f"{'Mesh Name':<30} {'Mesh Size':>12} {'#iter':>6} {'Speedup':>10} {'Max iter':>10}")
        print("-" * 80)
        
        for r in app_results:
            print(f"{r['mesh_name']:<30} {r['G_N']:>12} {r['num_iter']:>6} {format_speedup(r['speedup']):>10} {format_max_iter(r['max_iter']):>10}")
            all_results.append(r)
        
        # Print application summary if multiple meshes
        if len(app_results) > 1:
            avg_speedup = sum(r['speedup'] for r in app_results if not math.isinf(r['speedup'])) / len([r for r in app_results if not math.isinf(r['speedup'])])
            print("-" * 80)
            print(f"{'Average':<30} {'':<12} {'':<6} {format_speedup(avg_speedup):>10}")
    
    # Print combined table
    print("\n" + "=" * 80)
    print("Combined Table (All Applications)")
    print("=" * 80)
    print(f"\n{'App':<13} {'Setting':<8} {'Mesh Name':<25} {'Mesh Size':>12} {'#iter':>6} {'Speedup':>10} {'Max iter':>10}")
    print("-" * 95)
    
    for r in all_results:
        print(f"{r['app_name']:<13} {r['setting']:<8} {r['mesh_name']:<25} {r['G_N']:>12} {r['num_iter']:>6} {format_speedup(r['speedup']):>10} {format_max_iter(r['max_iter']):>10}")
    
    # Print LaTeX format
    print("\n" + "=" * 80)
    print("LaTeX Table Rows")
    print("=" * 80)
    print(generate_latex_table_rows(all_results))


if __name__ == "__main__":
    main()
