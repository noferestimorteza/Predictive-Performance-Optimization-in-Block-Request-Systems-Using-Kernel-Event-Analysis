import pandas as pd

# Detect the file encoding
with open('holedata.csv', 'rb') as f:
    raw_data = f.read(1000)  # Read the first 1000 bytes to detect encoding
    if raw_data.startswith(b'\xFF\xFE'):
        encoding = 'utf-16-le'
    elif raw_data.startswith(b'\xFE\xFF'):
        encoding = 'utf-16-be'
    else:
        encoding = 'utf-8'  # Fallback to utf-8 if no BOM is found

# Read the file with the detected encoding
df = pd.read_csv('holedata.csv', sep='\t', encoding=encoding)

# Clean the Timestamp column (remove extra data)
df['Timestamp'] = df['Timestamp'].str.split().str[0]

# Convert the Timestamp to datetime objects
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S.%f', errors='coerce')

# Filter the relevant events
block_rq_insert_events = df[df['Event type'] == 'block_rq_insert']
block_rq_issue_events = df[df['Event type'] == 'block_rq_issue']
block_rq_complete_events = df[df['Event type'] == 'block_rq_complete']

# Initialize a list to store the results
results = []

# Iterate over each block_rq_insert event
for _, insert_event in block_rq_insert_events.iterrows():
    tid = insert_event['TID']
    pid = insert_event['PID']
    
    # Find the corresponding block_rq_issue event with the same TID and PID
    issue_events = block_rq_issue_events[(block_rq_issue_events['TID'] == tid) & (block_rq_issue_events['PID'] == pid)]
    
    # Find the corresponding block_rq_complete event with the same TID and PID
    complete_events = block_rq_complete_events[(block_rq_complete_events['TID'] == tid) & (block_rq_complete_events['PID'] == pid)]
    
    # Ensure all events are present
    if not issue_events.empty and not complete_events.empty:
        # Get the closest block_rq_issue event after the block_rq_insert event
        issue_event = issue_events[issue_events['Timestamp'] >= insert_event['Timestamp']]
        if not issue_event.empty:
            issue_event = issue_event.iloc[0]
            
            # Get the closest block_rq_complete event after the block_rq_issue event
            complete_event = complete_events[complete_events['Timestamp'] >= issue_event['Timestamp']]
            if not complete_event.empty:
                complete_event = complete_event.iloc[0]

                # Calculate time differences
                insert_to_issue = (issue_event['Timestamp'] - insert_event['Timestamp']).total_seconds() * 1000
                issue_to_complete = (complete_event['Timestamp'] - issue_event['Timestamp']).total_seconds() * 1000

                # Store all event data in the results list
                results.append({
                    'PID': pid,
                    'TID': tid,
                    'block_rq_insert_data': insert_event.to_dict(),  # All data for block_rq_insert
                    'block_rq_issue_data': issue_event.to_dict(),    # All data for block_rq_issue
                    'block_rq_complete_data': complete_event.to_dict(),  # All data for block_rq_complete
                    'insert_to_issue': insert_to_issue,
                    'issue_to_complete': issue_to_complete
                })

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)

# Print the results DataFrame
print(results_df)

# Save the results to a file
with open('request_timestamps.txt', 'w') as f:
    # Write the header
    f.write("PID$$TID$$block_rq_insert_data$$block_rq_issue_data$$block_rq_complete_data$$insert_to_issue$$issue_to_complete\n")
    
    # Write each line of data
    for _, row in results_df.iterrows():
        f.write(f"{row['PID']}$${row['TID']}$${row['block_rq_insert_data']}$${row['block_rq_issue_data']}$${row['block_rq_complete_data']}$${row['insert_to_issue']}$${row['issue_to_complete']}\n")