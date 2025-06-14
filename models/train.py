import sys
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib # For saving the pipeline
import warnings
from sklearn.preprocessing import RobustScaler, PowerTransformer
from category_encoders import TargetEncoder
from sklearn.svm import SVC

# Suppress harmless warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
def preprocess_data(df):
    # Normalize Oui/Non columns to boolean
    bool_map = {'Oui': True, 'Non': False}
    for col in df.columns:
        if df[col].dropna().isin(bool_map.keys()).all():
            df[col] = df[col].map(bool_map)
    
    # Ensure proper dtypes
    bool_columns = ['compte_bancaire_actif', 'incident_bancaire', 'historique_credit']
    for col in bool_columns:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    return df

def train_and_evaluate(name_dataset:str,pipeline_filename):
    print("--- Starting ML Pipeline: Loan Approval Prediction (Professional Grade) ---\n")
    print("1. Loading Dataset...")
    # Load the dataset. Ensure the path is correct.
    try:
        dataset = pd.read_csv(name_dataset)
        dataset = preprocess_data(dataset)  # Preprocess the dataset

        print(f"Dataset loaded successfully. Shape: {dataset.shape}")
        print("Dataset head:\n", dataset.head())
    except FileNotFoundError:
        print("Error: Dataset CSV not found at 'train/financing_approval_balanced_dataset_Tunisia_with_interactions.csv'. Please check the path.")
        exit() # Exit if dataset isn't found

    # Separate features (X) and target (y)
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    

    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=[np.number, 'bool']).columns.tolist()

    categorical_features = [col for col in categorical_features if col not in numerical_features]
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('power_transform', PowerTransformer())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('target_enc', TargetEncoder())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
        ],
        remainder='passthrough'
    )
    model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42)) # Initial classifier (hyperparameters will be tuned)
    ])

   
    # Check if every class in y has at least 2 samples before stratified split
    class_counts = y.value_counts()
    if (class_counts < 2).any():
        print("Error: The least populated class in y has fewer than 2 members. Stratified split cannot proceed.")
        print("Class distribution:\n", class_counts)
        exit()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    param_grid = {
    'classifier__n_estimators': [100, 200, 300], # Number of trees in the forest
    'classifier__max_depth': [None, 10, 20],   # Maximum depth of the tree (None means unlimited)
    'classifier__min_samples_split': [2, 5, 10], # Minimum number of samples required to split an internal node
    'classifier__min_samples_leaf': [1, 2, 4]    # Minimum number of samples required to be at a leaf node
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='roc_auc',
        n_jobs=2,  # Use 2 parallel jobs for faster processing
        verbose=1
    )


    print("   Starting Grid Search (this may take a while)...")
    grid_search.fit(X_train, y_train) # Fit GridSearchCV on the training data
    print("   Grid Search complete.")

    # Get the best estimator found by GridSearchCV
    best_model = grid_search.best_estimator_
    print(f"   Best parameters found: {grid_search.best_params_}")
    print(f"   Best ROC-AUC score during cross-validation: {grid_search.best_score_:.4f}")

    print("\n6. Evaluating Best Model on the Held-Out Test Set...")
    # Make predictions on the unseen test set using the best model
    y_pred_best = best_model.predict(X_test)
    y_proba_best = best_model.predict_proba(X_test)[:, 1] # Probability of approval

    # Calculate evaluation metrics
    accuracy_best = accuracy_score(y_test, y_pred_best)
    roc_auc_best = roc_auc_score(y_test, y_proba_best)
    class_report_best = classification_report(y_test, y_pred_best)
    conf_matrix_best = confusion_matrix(y_test, y_pred_best)

    print(f"   Test Set Accuracy: {accuracy_best:.4f}")
    print(f"   Test Set ROC AUC: {roc_auc_best:.4f}")
    print("\n   Test Set Classification Report:\n", class_report_best)
    print("\n   Test Set Confusion Matrix:\n", conf_matrix_best)
    print("     (Row: Actual, Column: Predicted)")
    print("     [[True Negatives  False Positives]")
    print("      [False Negatives True Positives]]")


    print("\n7. Extracting Feature Importances...")
   
    feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()


    # Get feature importances from the RandomForestClassifier (the 'classifier' step in the pipeline)
    importances = best_model.named_steps['classifier'].feature_importances_

    # Create a Series for easier viewing and sorting
    feature_importances = pd.Series(importances, index=feature_names)

    # Sort features by importance
    sorted_importances = feature_importances.sort_values(ascending=False)

    print("   Top 10 Feature Importances (from the best model):")
    print(sorted_importances.head(10))

    print("\n8. Saving the Best Trained Model Pipeline...")
    # Save the entire best pipeline to a .pkl file.
    # This file contains the preprocessor and the best trained Random Forest model.
    joblib.dump(best_model, pipeline_filename)
    print(f"   Best model pipeline (including preprocessor) saved as '{pipeline_filename}'")

    print("\n9. Example Predictions (using the best model on test data, converted to likelihood %)...")
    # Apply the best model to a few test samples and display probabilities
    scores_out_of_100_best = y_proba_best * 100
    for i, score in enumerate(scores_out_of_100_best[:10]): # Display 10 examples
        actual_label = y_test.iloc[i] if isinstance(y_test, pd.Series) else y_test[i]
        print(f"   Client {i+1} (Actual: {actual_label}): Approval likelihood = {score:.2f}% (Predicted: {y_pred_best[i]})")

    print("\n--- ML Pipeline Execution Complete ---")

if __name__ == "__main__":

    train_and_evaluate(sys.argv[1], sys.argv[2])

