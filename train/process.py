import pandas as pd
from sklearn import preprocessing
dataset = pd.read_csv("financing_approval_balanced_dataset_Tunisia_with_interactions.csv")
# Convert categorical variables to numerical
categorical_columns = dataset.select_dtypes(include=['object']).columns
print("Categorical columns:", categorical_columns)
for column in categorical_columns:
    dataset[column] = preprocessing.LabelEncoder().fit_transform(dataset[column])
# Save the processed dataset
dataset.to_csv("financing_approval_balanced_dataset_Tunisia_with_interactions.csv", index=False)