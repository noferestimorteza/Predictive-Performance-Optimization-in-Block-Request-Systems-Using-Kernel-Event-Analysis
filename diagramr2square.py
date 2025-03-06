import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# R-squared scores for the models
data = {
    'Model': ['LR', 'RF', 'LightGBM', 'DNN'] * 2,
    'R2_Score': [ 0.23615761396747081,0.8596333254170476, 0.7585128349049569, 0.2298882007598877,0.23865786704588776, 0.9108893956743919, 0.7923294861683944, 0.23254454135894775],
    'Dataset': ['Block_rq_insert'] * 4 + ['Block_rq_insert + Block_rq_issue'] * 4
}

df = pd.DataFrame(data)

# Plotting with Seaborn
plt.figure(figsize=(10, 6))
ax = sns.barplot(y='Model', x='R2_Score', hue='Dataset', data=df, palette='gray', orient='h')

# Adding hatching for black-and-white differentiation
hatches = ['/', '/', '/', '/', '.', '.', '.', '.', '/', '.']

for i, bar in enumerate(ax.patches):
    bar.set_edgecolor('black')

# Adding labels and title
plt.xlabel('R-squared Score')
plt.ylabel('Model')

# Display the legend
plt.legend(title='Prediction Data')

# Grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.tight_layout()
plt.show()