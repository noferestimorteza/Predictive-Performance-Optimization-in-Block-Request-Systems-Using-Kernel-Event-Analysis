import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import time
import psutil

# Step 1: Load the data
data = pd.read_csv('extracted_data_insert_and_issue_complete_request_params.csv')

# Step 2: Handle missing values
data['insert_to_issue'] = data['insert_to_issue'].fillna(0)
data['issue_to_complete'] = data['issue_to_complete'].fillna(0)

# Step 3: Calculate total duration
data['total_duration'] = data['insert_to_issue'] + data['issue_to_complete']

# Step 4: Identify abnormal rows based on mean + 3*std threshold
threshold = data['total_duration'].mean() + 3 * data['total_duration'].std()
data['abnormal'] = (data['total_duration'] > threshold).astype(int)

# Step 5: Prepare features and target
X = data.drop(columns=['issue_to_complete', 'insert_to_issue', 'total_duration', 'abnormal'])
selected_features = ['brqinsrt_Channel', 'brqinsrt_CPU', 'brqinsrt_Event type', 'brqinsrt_Contents.dev', 'brqinsrt_Contents.sector', 
                     'brqinsrt_Contents.nr_sector', 'brqinsrt_Contents.bytes', 'brqinsrt_Contents.tid', 'brqinsrt_Contents.rwbs',
                     'brqinsrt_Contents.comm','brqinsrt_Contents.context.packet_seq_num','brqinsrt_Contents.context.cpu_id','brqinsrt_TPH.magic','brqinsrt_TPH.uuid','brqinsrt_TPH.stream_id','brqinsrt_TPH.stream_instance_id','brqinsrt_PH.content_size','brqinsrt_PH.packet_size','brqinsrt_PH.packet_seq_num','brqinsrt_PH.events_discarded','brqinsrt_PH.cpu_id','brqinsrt_PH.duration','brqinsrt_Stream_Context','brqinsrt_Event_Context','brqinsrt_TID','brqinsrt_Prio','brqinsrt_PID','brqinsrt_Source']
# selected_features = ['brqinsrt_Channel', 'brqinsrt_CPU', 'brqinsrt_Event type', 'brqinsrt_Contents.dev', 'brqinsrt_Contents.sector', 
#                      'brqinsrt_Contents.nr_sector', 'brqinsrt_Contents.bytes', 'brqinsrt_Contents.tid', 'brqinsrt_Contents.rwbs',
#                      'brqinsrt_Contents.comm','brqinsrt_Contents.context.packet_seq_num','brqinsrt_Contents.context.cpu_id','brqinsrt_TPH.magic','brqinsrt_TPH.uuid','brqinsrt_TPH.stream_id','brqinsrt_TPH.stream_instance_id','brqinsrt_PH.content_size','brqinsrt_PH.packet_size','brqinsrt_PH.packet_seq_num','brqinsrt_PH.events_discarded','brqinsrt_PH.cpu_id','brqinsrt_PH.duration','brqinsrt_Stream_Context','brqinsrt_Event_Context','brqinsrt_TID','brqinsrt_Prio','brqinsrt_PID','brqinsrt_Source',
# 					 'brqissue_Channel', 'brqissue_CPU', 'brqissue_Event type', 'brqissue_Contents.dev', 'brqissue_Contents.sector', 
#                      'brqissue_Contents.nr_sector', 'brqissue_Contents.bytes', 'brqissue_Contents.tid', 'brqissue_Contents.rwbs',
#                      'brqissue_Contents.comm','brqissue_Contents.context.packet_seq_num','brqissue_Contents.context.cpu_id','brqissue_TPH.magic','brqissue_TPH.uuid','brqissue_TPH.stream_id','brqissue_TPH.stream_instance_id','brqissue_PH.content_size','brqissue_PH.packet_size','brqissue_PH.packet_seq_num','brqissue_PH.events_discarded','brqissue_PH.cpu_id','brqissue_PH.duration','brqissue_Stream_Context','brqissue_Event_Context','brqissue_TID','brqissue_Prio','brqissue_PID','brqissue_Source']
X = X[selected_features]
y = data['abnormal']

# Step 6: Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Step 7: Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Step 8: Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Step 9: Bundle preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Step 10: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 11: Data preprocessing
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Step 12: Build the LightGBM Model
model = lgb.LGBMClassifier()

# Step 13: Train the LightGBM Model
start_train = time.time()
model.fit(X_train_preprocessed, y_train)
end_train = time.time()
training_time = end_train - start_train

# Step 14: Make predictions
start_time = time.time_ns()
y_pred_prob = model.predict_proba(X_test_preprocessed)[:, 1]
y_pred = (y_pred_prob > 0.5).astype(int)
end_time = time.time_ns()

# Inference overhead
inference_time = end_time - start_time
average_inference_time = inference_time / len(X_test_preprocessed)

# Step 15: CPU and Memory Overhead
process = psutil.Process()
cpu_before = process.cpu_percent(interval=None)
memory_before = process.memory_info().rss

_ = model.predict(X_test_preprocessed)

cpu_after = process.cpu_percent(interval=None)
memory_after = process.memory_info().rss

cpu_overhead = cpu_after - cpu_before
memory_overhead = (memory_after - memory_before) / (1024 ** 2)

# Step 16: Evaluation
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
accuracy = accuracy_score(y_test, y_pred)

# Results
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Training Time: {training_time:.6f} seconds")
print(f"Total Inference Time: {inference_time:.2f} ns")
print(f"Average Inference Time per Request: {average_inference_time:.2f} ns")
print(f"CPU Overhead: {cpu_overhead}%")
print(f"Memory Overhead: {memory_overhead:.2f} MB")

# Display abnormal rows
abnormal_rows = data[data['abnormal'] == 1]
print(f"Number of Abnormal Rows: {len(abnormal_rows)}")