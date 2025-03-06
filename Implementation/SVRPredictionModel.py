import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Step 1: Load the data
# Replace 'your_file.csv' with the path to your CSV file
data = pd.read_csv('extracted_data_insert_and_issue_complete_request_params.csv')

# Step 2: Separate features and target variables
X = data.drop(columns=['insert_to_issue', 'issue_to_complete'])
y = data[['insert_to_issue', 'issue_to_complete']]

# Step 3: Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Step 4: Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with the mean
    ('scaler', StandardScaler())  # Scale numerical features
])

# Step 5: Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with the most frequent value
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical variables
])

# Step 6: Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Step 7: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Define the SVR model
# Use SVR as the base regressor and MultiOutputRegressor for multi-target regression
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)  # You can tune these hyperparameters
model = MultiOutputRegressor(svr_model)

# Step 9: Create and evaluate the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Apply preprocessing
    ('model', model)  # Train the model
])

# Step 10: Train the model
pipeline.fit(X_train, y_train)

# Step 11: Evaluate the model
y_pred = pipeline.predict(X_test)

# Calculate R^2 score and RMSE
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5  # Calculate RMSE manually

print(f'Model R^2 score: {r2}')
print(f'Model RMSE: {rmse}')

# Step 12: Make predictions
# Convert predictions to a DataFrame for easier analysis
predictions_df = pd.DataFrame(y_pred, columns=['insert_to_issue_pred', 'issue_to_complete_pred'])

# Compare predictions with actual values
results = pd.concat([y_test.reset_index(drop=True), predictions_df], axis=1)
print(results.head())

# Step 13: Save the model
# Save the trained model to a file
joblib.dump(pipeline, 'svr_model.pkl')

# Step 14: Load the model (optional)
# Load the model from the file
loaded_model = joblib.load('svr_model.pkl')

# Make predictions with the loaded model
loaded_predictions = loaded_model.predict(X_test)
print(loaded_predictions[:5])  # Print the first 5 predictions