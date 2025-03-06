import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter
import numpy as np
# Load your data into a DataFrame
# Assuming your data is in a CSV file
df = pd.read_csv('extracted_data_insert_and_issue_complete_request_params.csv')

# Calculate insert_to_issue
# df['insert_to_issue'] = df['brqissue_Timestamp'] - df['brqinsrt_Timestamp']
df['brqinsrt_Timestamp2'] = pd.to_datetime(df['brqinsrt_Timestamp'], format='mixed', errors='coerce')
#df['brqinsrt_Timestamp2'] = pd.to_datetime(df['brqinsrt_Timestamp'])
df['brqinsrt_Timestamp3'] = df['brqinsrt_Timestamp2'] - pd.Timedelta(hours=7, minutes=30)
df['brqinsrt_HourMinute'] = df['brqinsrt_Timestamp3'].dt.strftime('%H:%M:%S.%f')

# Plot the data

# Extract time part from the timestamps (assuming the format is consistent)
time_only = df['brqinsrt_HourMinute'].str[:-3]  # Extract the last part (time) after splitting by space

plt.figure(figsize=(10, 6))
plt.bar(time_only, df['insert_to_issue'], color='b', label='Insert to Issue Duration', log=True)
plt.bar(time_only, df['issue_to_complete'], bottom=df['insert_to_issue'], color='r', label='Issue to Complete Duration', log=True)

plt.xlabel('Block_RQ_Insert Timestamp')
plt.ylabel('Event Durations')

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))  # Limit to 6 ticks

# Rotate labels and adjust layout
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend()  # Show legend to distinguish lines
plt.show()


# df['insert_to_complete'] = df['insert_to_issue'] + df['issue_to_complete']

# avg_insert_to_issue = df['insert_to_complete'].mean()
# std_insert_to_issue = df['insert_to_complete'].std()

# # Plot the histogram with a logarithmic y-axis
# plt.figure(figsize=(10, 6))
# plt.hist(df['insert_to_complete'], bins=50, color='green', alpha=0.7, log=True)
# #plt.title('Distribution of Insert to Complete Time (Log Scale on Y-Axis)')
# plt.xlabel('Insert to Complete Time (ms)')
# plt.ylabel('Frequency (Log Scale)')

# # Add vertical lines for average, average + std, and average + 3std
# plt.axvline(avg_insert_to_issue, color='red', linestyle='--', label=f'Average: {avg_insert_to_issue:.2f} ms')
# plt.axvline(avg_insert_to_issue + std_insert_to_issue, color='green', linestyle='-.', label=f'Average + Std: {avg_insert_to_issue + std_insert_to_issue:.2f} ms')
# plt.axvline(avg_insert_to_issue + 3 * std_insert_to_issue, color='purple', linestyle=':', label=f'Average + 3Std: {avg_insert_to_issue + 3 * std_insert_to_issue:.2f} ms')

# #plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
# plt.ticklabel_format(axis='x', style='plain')  # Use plain formatting for x-axis


# # Add legend and grid
# plt.legend()
# plt.grid(True)
# plt.show()

# df['insert_to_complete'] = df['insert_to_issue'] + df['issue_to_complete']

# # Calculate average and standard deviation
# avg_insert_to_issue = df['insert_to_complete'].mean()
# std_insert_to_issue = df['insert_to_complete'].std()

# # Categorize the events
# less_than_avg = df[df['insert_to_complete'] < avg_insert_to_issue].shape[0]
# between_avg_std = df[(df['insert_to_complete'] >= avg_insert_to_issue) & (df['insert_to_complete'] < avg_insert_to_issue + std_insert_to_issue)].shape[0]
# between_std_3std = df[(df['insert_to_complete'] >= avg_insert_to_issue + std_insert_to_issue) & (df['insert_to_complete'] < avg_insert_to_issue + 3 * std_insert_to_issue)].shape[0]
# more_than_3std = df[df['insert_to_complete'] >= avg_insert_to_issue + 3 * std_insert_to_issue].shape[0]

# # Create labels and sizes for the pie chart
# labels = [
#     'Less than Average',
#     'Average < Time <= Average + Std',
#     'Average + Std < Time <= Average + 3Std',
#     'More than Average + 3Std'
# ]
# sizes = [less_than_avg, between_avg_std, between_std_3std, more_than_3std]

# # Plot the pie chart
# plt.figure(figsize=(8, 8))
# plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightgreen', 'orange', 'lightcoral'])
# # plt.title('Distribution of Insert to Complete Times')
# plt.show()