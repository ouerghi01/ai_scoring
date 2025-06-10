import pandas as pd

# Load dataset with generic names like Attribute1, Attribute2...
df = pd.read_csv("german_credit_data.csv")
column_mapping = {
    "Attribute1": "checking_status",
    "Attribute2": "duration",
    "Attribute3": "credit_history",
    "Attribute4": "purpose",
    "Attribute5": "amount",
    "Attribute6": "savings",
    "Attribute7": "employment_since",
    "Attribute8": "installment_rate",
    "Attribute9": "personal_status",
    "Attribute10": "guarantor",
    "Attribute11": "residence_since",
    "Attribute12": "property",
    "Attribute13": "age",
    "Attribute14": "installment_plan",
    "Attribute15": "housing",
    "Attribute16": "existing_credits",
    "Attribute17": "job",
    "Attribute18": "people_liable",
    "Attribute19": "telephone",
    "Attribute20": "foreign_worker"
    # Add your target column if needed, e.g. "Class": "credit_risk"
}

# Apply renaming
df = df.rename(columns=column_mapping)

# Save to a new CSV
df.to_csv("german_credit_data_renamed.csv", index=False)

print("✅ Renamed and saved to 'german_credit_data_renamed.csv'")