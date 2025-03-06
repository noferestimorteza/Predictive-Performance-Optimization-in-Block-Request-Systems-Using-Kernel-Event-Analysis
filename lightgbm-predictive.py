import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
import time
import psutil
from lightgbm import LGBMRegressor

# Step 1: Load the data
data = pd.read_csv('extracted_data_insert_and_issue_complete_request_params.csv')

# Step 2: Separate features and target variables
data['insert_to_issue'] = data['insert_to_issue'].fillna(0)
data['issue_to_complete'] = data['issue_to_complete'].fillna(0)

y = (data['insert_to_issue'] + data['issue_to_complete']).to_frame(name='total_duration')

X = data.drop(columns=['issue_to_complete', 'insert_to_issue'])
# selected_features = ['brqinsrt_Channel', 'brqinsrt_CPU', 'brqinsrt_Event type', 'brqinsrt_Contents.dev', 'brqinsrt_Contents.sector', 
#                      'brqinsrt_Contents.nr_sector', 'brqinsrt_Contents.bytes', 'brqinsrt_Contents.tid', 'brqinsrt_Contents.rwbs',
#                      'brqinsrt_Contents.comm','brqinsrt_Contents.context.packet_seq_num','brqinsrt_Contents.context.cpu_id','brqinsrt_TPH.magic','brqinsrt_TPH.uuid','brqinsrt_TPH.stream_id','brqinsrt_TPH.stream_instance_id','brqinsrt_PH.content_size','brqinsrt_PH.packet_size','brqinsrt_PH.packet_seq_num','brqinsrt_PH.events_discarded','brqinsrt_PH.cpu_id','brqinsrt_PH.duration','brqinsrt_Stream_Context','brqinsrt_Event_Context','brqinsrt_TID','brqinsrt_Prio','brqinsrt_PID','brqinsrt_Source']
selected_features = ['brqinsrt_Channel', 'brqinsrt_CPU', 'brqinsrt_Event type', 'brqinsrt_Contents.dev', 'brqinsrt_Contents.sector', 
                     'brqinsrt_Contents.nr_sector', 'brqinsrt_Contents.bytes', 'brqinsrt_Contents.tid', 'brqinsrt_Contents.rwbs',
                     'brqinsrt_Contents.comm','brqinsrt_Contents.context.packet_seq_num','brqinsrt_Contents.context.cpu_id','brqinsrt_TPH.magic','brqinsrt_TPH.uuid','brqinsrt_TPH.stream_id','brqinsrt_TPH.stream_instance_id','brqinsrt_PH.content_size','brqinsrt_PH.packet_size','brqinsrt_PH.packet_seq_num','brqinsrt_PH.events_discarded','brqinsrt_PH.cpu_id','brqinsrt_PH.duration','brqinsrt_Stream_Context','brqinsrt_Event_Context','brqinsrt_TID','brqinsrt_Prio','brqinsrt_PID','brqinsrt_Source',
					 'brqissue_Channel', 'brqissue_CPU', 'brqissue_Event type', 'brqissue_Contents.dev', 'brqissue_Contents.sector', 
                     'brqissue_Contents.nr_sector', 'brqissue_Contents.bytes', 'brqissue_Contents.tid', 'brqissue_Contents.rwbs',
                     'brqissue_Contents.comm','brqissue_Contents.context.packet_seq_num','brqissue_Contents.context.cpu_id','brqissue_TPH.magic','brqissue_TPH.uuid','brqissue_TPH.stream_id','brqissue_TPH.stream_instance_id','brqissue_PH.content_size','brqissue_PH.packet_size','brqissue_PH.packet_seq_num','brqissue_PH.events_discarded','brqissue_PH.cpu_id','brqissue_PH.duration','brqissue_Stream_Context','brqissue_Event_Context','brqissue_TID','brqissue_Prio','brqissue_PID','brqissue_Source']
X = X[selected_features]


# Step 3: Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Step 4: Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Step 5: Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Step 6: Bundle preprocessing
preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Step 7: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Preprocess the data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Step 9: Build the LightGBM model
lgbm_model = LGBMRegressor(n_estimators=100, random_state=42)
lgbm_model.fit(X_train_preprocessed, y_train.values.ravel())

# Step 10: Predict on the test set
y_pred = lgbm_model.predict(X_test_preprocessed)

# Step 11: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("R-squared:", r2)

# Inference Time
start_time = time.time_ns()
y_pred = lgbm_model.predict(X_test_preprocessed)
end_time = time.time_ns()

inference_time = end_time - start_time
average_inference_time = inference_time / len(X_test_preprocessed)

print(f"Total Inference Time: {inference_time:.2f} ns")
print(f"Average Inference Time per Request: {average_inference_time:.2f} ns")

# Resource Overhead
process = psutil.Process()
cpu_before = process.cpu_percent(interval=None)
memory_before = process.memory_info().rss

_ = lgbm_model.predict(X_test_preprocessed)

cpu_after = process.cpu_percent(interval=None)
memory_after = process.memory_info().rss

cpu_overhead = cpu_after - cpu_before
memory_overhead = (memory_after - memory_before) / (1024 ** 2)

print(f"CPU Overhead: {cpu_overhead}%")
print(f"Memory Overhead: {memory_overhead:.2f} MB")

# Training Time
start_train = time.time()
lgbm_model.fit(X_train_preprocessed, y_train.values.ravel())
end_train = time.time()

training_time = end_train - start_train
print(f"Training Time: {training_time:.6f} seconds")
