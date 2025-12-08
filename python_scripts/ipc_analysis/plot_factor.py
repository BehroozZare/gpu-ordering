import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('/home/behrooz/Desktop/Last_Project/gpu_ordering/python_scripts/ipc_analysis/IPC.csv')

# Create x-axis as entry indices
x = range(len(df))

# Extract the two columns for plotting
factor_metis = df['factor/metis factor']
org_factor_metis = df['org factor/metis factor']

# Create the plot
plt.figure(figsize=(12, 6))

plt.plot(x, factor_metis, label='patch ordering / metis factor', alpha=0.7)
plt.plot(x, org_factor_metis, label='lagging to metis factor', alpha=0.7)

plt.xlabel('Entry')
plt.ylabel('Normalized to Metis')
plt.title('Factor Ratios Normalized to Metis')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/behrooz/Desktop/Last_Project/gpu_ordering/python_scripts/ipc_analysis/factor_plot.png', dpi=150)
plt.show()
