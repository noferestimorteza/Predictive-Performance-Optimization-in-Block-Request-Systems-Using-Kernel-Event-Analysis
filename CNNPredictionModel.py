import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

# Step 1: Load the data
# Replace 'your_file.csv' with the path to your CSV file
data = pd.read_csv('extracted_data_insert_and_issue_complete_request_params.csv')

# Step 2: Separate features and target variables
X = data.drop(columns=['issue_to_complete'])
y = data[[ 'issue_to_complete']]

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

# Step 8: Preprocess the data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Step 9: Define the Neural Network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_preprocessed.shape[1],)),  # Input layer
    Dropout(0.2),  # Dropout for regularization
    Dense(64, activation='relu'),  # Hidden layer
    Dropout(0.2),  # Dropout for regularization
    Dense(32, activation='relu'),  # Hidden layer
    Dense(2)  # Output layer (2 neurons for 2 target variables)
])

# Step 10: Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 11: Train the model
history = model.fit(
    X_train_preprocessed, y_train,
    epochs=50,  # Number of epochs
    batch_size=32,  # Batch size
    validation_split=0.2,  # Validation split
    verbose=1
)

# Step 12: Evaluate the model
y_pred = model.predict(X_test_preprocessed)

# Calculate R^2 score and RMSE
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5  # Calculate RMSE manually

print(f'Model R^2 score: {r2}')
print(f'Model RMSE: {rmse}')

# Step 13: Make predictions
# Convert predictions to a DataFrame for easier analysis
predictions_df = pd.DataFrame(y_pred, columns=[ 'issue_to_complete_pred'])

# Compare predictions with actual values
results = pd.concat([y_test.reset_index(drop=True), predictions_df], axis=1)
print(results.head())

# Step 14: Save the model
# Save the trained model to a file
model.save('neural_network_model.h5')

# Step 15: Load the model (optional)
# Load the model from the file
loaded_model = tf.keras.models.load_model('neural_network_model.h5')

# Make predictions with the loaded model
loaded_predictions = loaded_model.predict(X_test_preprocessed)
print(loaded_predictions[:5])  # Print the first 5 predictions