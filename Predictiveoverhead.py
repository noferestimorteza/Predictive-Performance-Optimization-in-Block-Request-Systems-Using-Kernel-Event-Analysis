import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['LR', 'RF', 'LightGBM', 'DNN']

# Average Inference Time (nanoseconds)
inference_time_insert = [0.022116 ,6.762303, 0.075229, 30.525294]
inference_time_insert_issue = [0.058798, 12.677802, 0.128970, 31.184986]

# Bar width and positions
bar_width = 0.35
index = np.arange(len(models))

# Plotting
plt.figure(figsize=(10, 6))

# Bars for block_rq_insert
plt.bar(index, inference_time_insert, bar_width, label='block_rq_insert', color='red')

# Bars for block_rq_insert + block_rq_issue
plt.bar(index + bar_width, inference_time_insert_issue, bar_width, label='block_rq_insert + block_rq_issue', color='orange')

# Labels and title
plt.xlabel('Models')
plt.ylabel('Traning Time (seconds)')
plt.xticks(index + bar_width / 2, models)  # Center x-ticks
plt.yscale('log')  # Use log scale for better visualization
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

# Show plot
plt.tight_layout()
plt.show()