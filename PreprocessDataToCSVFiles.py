import re
import pandas as pd
from datetime import datetime

# Function to extract fields from a line of block_rq_insert_data
def extract_fields(line):
    try:
        # Convert the line to a dictionary
        line = line.replace("Timestamp('", "'").replace("')", "'")  # Remove Timestamp wrapper
        line = line.replace(" nan", " None")  # Replace 'nan' with 'None'
        data = eval(line)
       # Extract Contents
        contents = data['Contents']
        # Remove trailing commas and split into key-value pairs
        contents_cleaned = re.sub(r',\s*$', '', contents)  # Remove trailing comma
        contents_dict = dict(re.findall(r'(\w+)=("[^"]+"|\S+)', contents_cleaned))
        
        # Extract Trace Packet Header (TPH)
        tph = data['Trace Packet Header']
        
        # Use a regex to capture both bracketed and non-bracketed values
        tph_matches = re.findall(r'(\w+)=(?:\[([^\]]+)\]|([^,]+))', tph)
        tph_dict = {}
        for match in tph_matches:
            key = match[0]
            value = match[1] if match[1] else match[2]  # Use the non-empty value
            tph_dict[key] = value
                
        # Convert uuid to a list of integers
        if 'uuid' in tph_dict:
            tph_dict['uuid'] = [int(x) for x in tph_dict['uuid'].split(', ')]

        # Extract Packet Context (PH)
        ph = data['Packet Context']
        ph_dict = dict(re.findall(r'(\w+)=(\d+)', ph))
        
        # Calculate duration
        duration = int(ph_dict['timestamp_end']) - int(ph_dict['timestamp_begin'])
        
        # Create the final dictionary with all fields
        result = {
            'Timestamp': data['Timestamp'],
            'Channel': data['Channel'],
            'CPU': data['CPU'],
            'Event type': data['Event type'],
            'Contents.dev': int(contents_dict['dev'].replace(',','')),
            'Contents.sector': int(contents_dict['sector'].replace(',','')),
            'Contents.nr_sector': int(contents_dict['nr_sector'].replace(',','')),
            'Contents.bytes': int(contents_dict['bytes'].replace(',','')),
            'Contents.tid': int(contents_dict['tid'].replace(',','')),
            'Contents.rwbs': int(contents_dict['rwbs'].replace(',','')),
            'Contents.comm': contents_dict['comm'].strip('"'),
            'Contents.context.packet_seq_num': int(contents_dict['packet_seq_num'].replace(',','')),
            'Contents.context.cpu_id': int(contents_dict['cpu_id'].replace(',','')),
            'TPH.magic': int(tph_dict['magic'].replace(',','')),
            'TPH.uuid': tph_dict['uuid'],
            'TPH.stream_id': int(tph_dict['stream_id'].replace(',','')),
            'TPH.stream_instance_id': int(tph_dict['stream_instance_id'].replace(']','')),
            'PH.content_size': int(ph_dict['content_size'].replace(',','')),
            'PH.packet_size': int(ph_dict['packet_size'].replace(',','')),
            'PH.packet_seq_num': int(ph_dict['packet_seq_num'].replace(',','')),
            'PH.events_discarded': int(ph_dict['events_discarded'].replace(',','')),
            'PH.cpu_id': int(ph_dict['cpu_id'].replace(',','')),
            'PH.duration': duration,
            'Stream_Context': data['Stream Context'],
            'Event_Context': data['Event Context'],
            'TID': data['TID'],
            'Prio': data['Prio'],
            'PID': data['PID'],
            'Source': data['Source']
        }
        
        return result
    except Exception as e:
        print(f"Error parsing line: {line}")
        print(f"Error: {e}")
        return None
    

def extract_fields_complete(line):
    try:
        # Convert the line to a dictionary
        line = line.replace("Timestamp('", "'").replace("')", "'")  # Remove Timestamp wrapper
        line = line.replace(" nan", " None")  # Replace 'nan' with 'None'
        print(line)
        data = eval(line)
        # Extract Contents
        contents = data['Contents']
        # Remove trailing commas and split into key-value pairs
        contents_cleaned = re.sub(r',\s*$', '', contents)  # Remove trailing comma
        contents_dict = dict(re.findall(r'(\w+)=("[^"]+"|\S+)', contents_cleaned))
        
        # Extract Trace Packet Header (TPH)
        tph = data['Trace Packet Header']
        
        # Use a regex to capture both bracketed and non-bracketed values
        tph_matches = re.findall(r'(\w+)=(?:\[([^\]]+)\]|([^,]+))', tph)
        tph_dict = {}
        for match in tph_matches:
            key = match[0]
            value = match[1] if match[1] else match[2]  # Use the non-empty value
            tph_dict[key] = value
                
        # Convert uuid to a list of integers
        if 'uuid' in tph_dict:
            tph_dict['uuid'] = [int(x) for x in tph_dict['uuid'].split(', ')]

        # Extract Packet Context (PH)
        ph = data['Packet Context']
        ph_dict = dict(re.findall(r'(\w+)=(\d+)', ph))
        
        # Calculate duration
        duration = int(ph_dict['timestamp_end']) - int(ph_dict['timestamp_begin'])
        
        # Create the final dictionary with all fields
        result = {
            'Timestamp': data['Timestamp'],
            'Channel': data['Channel'],
            'CPU': data['CPU'],
            'Event type': data['Event type'],
            'Contents.dev': int(contents_dict['dev'].replace(',','')),
            'Contents.sector': int(contents_dict['sector'].replace(',','')),
            'Contents.nr_sector': int(contents_dict['nr_sector'].replace(',','')),
            'Contents.error': int(contents_dict['error'].replace(',','')),
            'Contents.rwbs': int(contents_dict['rwbs'].replace(',','')),
            'Contents.context.packet_seq_num': int(contents_dict['packet_seq_num'].replace(',','')),
            'Contents.context.cpu_id': int(contents_dict['cpu_id'].replace(',','')),
            'TPH.magic': int(tph_dict['magic'].replace(',','')),
            'TPH.uuid': tph_dict['uuid'],
            'TPH.stream_id': int(tph_dict['stream_id'].replace(',','')),
            'TPH.stream_instance_id': int(tph_dict['stream_instance_id'].replace(']','')),
            'PH.content_size': int(ph_dict['content_size'].replace(',','')),
            'PH.packet_size': int(ph_dict['packet_size'].replace(',','')),
            'PH.packet_seq_num': int(ph_dict['packet_seq_num'].replace(',','')),
            'PH.events_discarded': int(ph_dict['events_discarded'].replace(',','')),
            'PH.cpu_id': int(ph_dict['cpu_id'].replace(',','')),
            'PH.duration': duration,
            'Stream_Context': data['Stream Context'],
            'Event_Context': data['Event Context'],
            'TID': data['TID'],
            'Prio': data['Prio'],
            'PID': data['PID'],
            'Source': data['Source']
        }
        
        return result
    except Exception as e:
        print(f"Error parsing line: {line}")
        print(f"Error: {e}")
        return None
    
csv_file = 'request_timestamps.txt'  # Replace with your CSV file path
df = pd.read_csv(csv_file, sep=r'\$\$', engine='python')  # Use $$ as the separator

# Initialize a list to store extracted data
extracted_data_list = []

# Process each row in the DataFrame
for index, row in df.iterrows():
    # Extract the block_rq_insert_data, block_rq_issue_data, and block_rq_complete_data fields
    block_rq_insert_data = row['block_rq_insert_data']
    block_rq_issue_data = row['block_rq_issue_data']
    block_rq_complete_data = row['block_rq_complete_data']

    extracted_insert_data = extract_fields(block_rq_insert_data)
    extracted_issue_data = extract_fields(block_rq_issue_data)
    extracted_complete_data = extract_fields_complete(block_rq_complete_data)

# Add prefixes to keys
    prefixed_insert_data = {f'brqinsrt_{key}': value for key, value in extracted_insert_data.items()}
    prefixed_issue_data = {f'brqissue_{key}': value for key, value in extracted_issue_data.items()}
    prefixed_complete_data = {f'brqcomplete_{key}': value for key, value in extracted_complete_data.items()}
    
    combined_data = {**prefixed_insert_data, **prefixed_issue_data, **prefixed_complete_data}
    # Extract fields from each data field
    if combined_data:
        # Add the PID and TID from the original row
        combined_data['PID'] = row['PID']
        combined_data['TID'] = row['TID']
        combined_data['insert_to_issue'] = row['insert_to_issue']
        combined_data['issue_to_complete'] = row['issue_to_complete']
        extracted_data_list.append(combined_data)

# Convert the list of dictionaries to a DataFrame
extracted_df = pd.DataFrame(extracted_data_list)

# Save the DataFrame to a CSV file (optional)
extracted_df.to_csv('extracted_data_insert_and_issue_complete_request_params.csv', index=False)

# Print the first few rows of the DataFrame
print(extracted_df.head())