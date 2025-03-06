import matplotlib.pyplot as plt
import pandas as pd

# ROC-AUC scores for the models
models = ['LR', 'RF', 'LightGBM', 'DNN']
insert_scores = [0.9173, 0.9929, 0.9962, 0.9411]
insert_issue_scores = [0.9208, 0.9979, 0.9970, 0.9471]

x = range(len(models))

plt.figure(figsize=(8, 6))
plt.bar(x, insert_scores, color='none', edgecolor='black', linewidth=1.1, label='Block_rq_insert', hatch='-', width=0.1)
plt.bar([i + 0.35 for i in x], insert_issue_scores, color='none', edgecolor='black', linewidth=1.1, label='Block_rq_insert + Block_rq_issue', width=0.1)

plt.xlabel('Anomaly Prediction Models')
plt.ylabel('ROC-AUC Score')
# plt.title('Comparison of ROC-AUC Scores for Different Models')
plt.xticks([i + 0.175 for i in x], models)

for i, score in enumerate(insert_scores):
    plt.text(i, score + 0.001, f'{score:.4f}', ha='center', va='bottom', fontsize=8)
for i, score in enumerate(insert_issue_scores):
    plt.text(i + 0.35, score + 0.001, f'{score:.4f}', ha='center', va='bottom', fontsize=8)

plt.legend(title='Dataset')
# plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
