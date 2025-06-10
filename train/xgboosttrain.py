import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
dataset = pd.read_csv("german_credit_data_processed.csv")
X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
y = y.replace({2: 1, 1: 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier()
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]  # Get probability of approval

# Convert to score out of 100
scores_out_of_100 = probs * 100

# Example output
for i, score in enumerate(scores_out_of_100[:5]):
    print(f"Client {i+1}: Approval likelihood = {score:.2f}%")
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# Save the model
model.save_model("xgboost_model.json")