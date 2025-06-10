from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch the dataset
statlog_german_credit_data = fetch_ucirepo(id=144)

# Extract features and targets
X = statlog_german_credit_data.data.features
y = statlog_german_credit_data.data.targets

# Combine features and target into a single DataFrame
df = pd.concat([X, y], axis=1)

# Save to CSV
df.to_csv("german_credit_data.csv", index=False)

print("✅ Saved to german_credit_data.csv")
