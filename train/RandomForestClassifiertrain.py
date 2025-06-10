import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # For saving the pipeline

print("--- Starting Model Training and Pipeline Creation ---")

# Load the dataset
dataset = pd.read_csv("train/financing_approval_balanced_dataset_Tunisia_with_interactions.csv")

# Separate features (X) and target (y)
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Define categorical and numerical features
categorical_features = ['Professional_Status', 'Sector', 'Existing_Loan']
# Documents_Complete is boolean, which can sometimes be handled by OHE directly,
# but it's good practice to list it if you expect it to be treated as a category.
# For simplicity, let's treat it as a categorical string for OneHotEncoder.
categorical_features.append('Documents_Complete')

numerical_features = [
    'Total_Acquisition_Price_DT',
    'Repayment_Duration_Years',
    'Monthly_Payment_DT',
    'Number_of_Clicks',
    'Time_Spent_Seconds'
]

# Create preprocessing steps
# One-Hot Encode categorical features (handle_unknown='ignore' is important for new unseen categories)
# Standard Scale numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Pass through any other columns not explicitly listed (none in this case)
)

# Create a pipeline that first preprocesses the data and then applies the RandomForestClassifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nFitting the model pipeline...")
# Train the entire pipeline (preprocessing and classification)
model_pipeline.fit(X_train, y_train)
print("Model pipeline training complete.")

# Make predictions and calculate probabilities on the test set
y_pred = model_pipeline.predict(X_test)
probs = model_pipeline.predict_proba(X_test)[:, 1] # Get probability of approval (class 1)

# Convert to score out of 100
scores_out_of_100 = probs * 100

print("\n--- Model Performance on Test Set ---")
print("Example Predictions (first 5 clients):")
for i, score in enumerate(scores_out_of_100[:5]):
    print(f"Client {i+1}: Approval likelihood = {score:.2f}%")

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the entire pipeline (preprocessor + classifier)
pipeline_filename = "random_forest_pipeline.pkl"
joblib.dump(model_pipeline, pipeline_filename)
print(f"\nFull model pipeline (including preprocessor) saved as '{pipeline_filename}'")