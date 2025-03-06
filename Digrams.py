import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data
models = ['Linear Regression (LR)', 'Random Forest (RF)', 'LightGBM', 'Deep Neural Network (DNN)']
block_rq_insert = [54878.29, 13753.20, 27461.51, 42823.91]
block_rq_insert_issue = [54866.92, 9253.47, 25019.87, 41933.38]

# Create DataFrame for plotting
data = pd.DataFrame({
    'Model': models * 2,
    'Value': block_rq_insert + block_rq_insert_issue,
    'Category': ['Block_rq_insert'] * 4 + ['Block_rq_insert + Block_rq_issue'] * 4
})

# Plotting
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Professional barplot
ax = sns.barplot(x='Category', y='Value', hue='Model', data=data, palette="Set2")

# Adding annotations
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', padding=3)

# Titles and labels
plt.xlabel('Prediction Info', fontsize=14)
plt.ylabel('MAE (ns)', fontsize=14)
plt.legend(title='Models', title_fontsize='13', fontsize='11')

# Final layout adjustments
plt.tight_layout()
plt.show()