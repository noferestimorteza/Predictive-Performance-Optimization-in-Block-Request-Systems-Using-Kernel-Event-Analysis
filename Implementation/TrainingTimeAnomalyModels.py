import matplotlib.pyplot as plt
import numpy as np

# Training times data
models = ['LR', 'RF', 'LightGBM', 'DNN']
insert_times = [0.053, 0.701, 0.185, 34.542]
insert_issue_times = [0.257, 0.863, 0.238, 34.300]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, insert_times, width, label='Block_rq_insert', color='gray', edgecolor='black')
rects2 = ax.bar(x + width/2, insert_issue_times, width, label='Block_rq_insert + Block_rq_issue', color='white', edgecolor='black', hatch='//')

ax.set_ylabel('Training Time (seconds)')
# ax.set_title('Training Time Comparison Across Models')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

for rect in rects1:
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(), f'{rect.get_height():.3f}', ha='center', va='bottom', fontsize=8)
for rect in rects2:
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(), f'{rect.get_height():.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()
