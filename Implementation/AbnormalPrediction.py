import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import re

# Load data
data = pd.read_csv('extracted_data_insert_and_issue_complete_request_params.csv')

# Define target variable (abnormal = 1, normal = 0)
threshold = data['issue_to_complete'].quantile(0.95)  # 95th percentile as threshold
data['abnormal'] = (data['issue_to_complete'] > threshold).astype(int)

# Separate features and target
X = data.drop(columns=['issue_to_complete', 'abnormal'])
y = data['abnormal']

selected_features = ['brqinsrt_Channel', 'brqinsrt_CPU', 'brqinsrt_Event type', 'brqinsrt_Contents.dev', 'brqinsrt_Contents.sector', 
                     'brqinsrt_Contents.nr_sector', 'brqinsrt_Contents.bytes', 'brqinsrt_Contents.tid', 'brqinsrt_Contents.rwbs',
                     'brqinsrt_Contents.comm','brqinsrt_Contents.context.packet_seq_num','brqinsrt_Contents.context.cpu_id','brqinsrt_TPH.magic','brqinsrt_TPH.uuid','brqinsrt_TPH.stream_id','brqinsrt_TPH.stream_instance_id','brqinsrt_PH.content_size','brqinsrt_PH.packet_size','brqinsrt_PH.packet_seq_num','brqinsrt_PH.events_discarded','brqinsrt_PH.cpu_id','brqinsrt_PH.duration','brqinsrt_Stream_Context','brqinsrt_Event_Context','brqinsrt_TID','brqinsrt_Prio','brqinsrt_PID','brqinsrt_Source']
X = X[selected_features]


# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing
# One-hot encode categorical columns
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_cat_encoded = encoder.fit_transform(X[categorical_cols])
cat_feature_names = encoder.get_feature_names_out(categorical_cols)
X_cat_encoded = pd.DataFrame(X_cat_encoded, columns=cat_feature_names)

# Scale numerical columns
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X[numerical_cols])
X_num_scaled = pd.DataFrame(X_num_scaled, columns=numerical_cols)

# Combine numerical and categorical features
X_encoded = pd.concat([X_num_scaled, X_cat_encoded], axis=1)

# Clean feature names
X_encoded.columns = [re.sub(r'[\[\]<>]', '', col) for col in X_encoded.columns]
X_encoded.columns = X_encoded.columns.astype(str)

# Verify cleaned feature names
print("Cleaned feature names:", X_encoded.columns)


# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier(random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate performance
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))