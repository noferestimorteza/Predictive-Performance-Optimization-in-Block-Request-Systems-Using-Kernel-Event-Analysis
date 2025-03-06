import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['LR', 'RF', 'LightGBM', 'DNN']
block_rq_insert = [1960.13, 5606.94, 3120.31, 44062.27]
block_rq_insert_issue = [489.16, 8829.52, 760.25, 42620.44]

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Set the width of the bars
bar_width = 0.35

# Set the positions of the bars on the x-axis
indices = np.arange(len(models))

# Plot the bars with patterns
bar1 = ax.bar(indices - bar_width/2, block_rq_insert, 0.3*bar_width, label='block_rq_insert', hatch='//', edgecolor='black', fill=False)
bar2 = ax.bar(indices + bar_width/2, block_rq_insert_issue, 0.3*bar_width, label='block_rq_insert + block_rq_issue', edgecolor='black', fill=False)

# Add labels, title, and legend
ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Average Inference Time per Request (ns)', fontsize=12)
# ax.set_title('Average Inference Time per Request', fontsize=14, fontweight='bold')
ax.set_xticks(indices)
ax.set_xticklabels(models, fontsize=12)
ax.legend(fontsize=12)

# Add value labels on top of the bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=10)

add_labels(bar1)
add_labels(bar2)

# Adjust y-axis to log scale for better readability
ax.set_yscale('log')

# Remove spines (top and right)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add gridlines for better readability
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.savefig('inference_time_comparison_bw.png', dpi=300, bbox_inches='tight')  # Save as high-resolution PNG
plt.show()