import pandas as pd
import numpy as np

# Set a seed for reproducibility to ensure the same dataset is generated each time
np.random.seed(42)

# Number of samples to generate in the initial pool (much larger than final dataset size)
num_samples_pool = 15000 # Increased pool size to ensure ample samples after adding new features

# Desired number of samples for the balanced dataset (e.g., 1000 approved, 1000 not approved)
desired_samples_per_class = 1000

# Generate synthetic data for the features
data = {
    'Professional_Status': np.random.choice(['Salarié', 'Salarié Et Professionnel À Usage Privé / Rentier', 'Professionnel À Usage Privé'], num_samples_pool, p=[0.6, 0.2, 0.2]),
    'Sector': np.random.choice(['Secteur Privé', 'Secteur Public'], num_samples_pool, p=[0.7, 0.3]),
    'Existing_Loan': np.random.choice(['Oui', 'Non'], num_samples_pool, p=[0.3, 0.7]),
    'Total_Acquisition_Price_DT': np.round(np.random.normal(loc=45000, scale=18000, size=num_samples_pool), 0),
    'Repayment_Duration_Years': np.random.choice([3, 4, 5, 6, 7], num_samples_pool, p=[0.1, 0.3, 0.3, 0.2, 0.1]),
    'Monthly_Payment_DT': np.round(np.random.normal(loc=1000, scale=350, size=num_samples_pool), 0),
    'Documents_Complete': np.random.choice([True, False], num_samples_pool, p=[0.85, 0.15]),

    # New interaction features
    'Number_of_Clicks': np.round(np.random.normal(loc=50, scale=20, size=num_samples_pool)).astype(int), # Average 50 clicks
    'Time_Spent_Seconds': np.round(np.random.normal(loc=300, scale=120, size=num_samples_pool)).astype(int), # Average 5 minutes (300 seconds)
    'Approved': np.nan # Placeholder for the target variable
}

df_pool = pd.DataFrame(data)

# Ensure numerical features are non-negative and within reasonable bounds
df_pool['Total_Acquisition_Price_DT'] = df_pool['Total_Acquisition_Price_DT'].apply(lambda x: max(8000, min(120000, x)))
df_pool['Monthly_Payment_DT'] = df_pool['Monthly_Payment_DT'].apply(lambda x: max(250, min(3500, x)))
df_pool['Number_of_Clicks'] = df_pool['Number_of_Clicks'].apply(lambda x: max(5, x)) # Minimum 5 clicks
df_pool['Time_Spent_Seconds'] = df_pool['Time_Spent_Seconds'].apply(lambda x: max(10, x)) # Minimum 10 seconds

# Simulate higher engagement for those with complete documents
# This makes the dataset more realistic as serious applicants typically complete documents and interact more
df_pool.loc[df_pool['Documents_Complete'] == True, 'Number_of_Clicks'] = df_pool.loc[df_pool['Documents_Complete'] == True, 'Number_of_Clicks'].apply(lambda x: np.random.normal(loc=60, scale=25)).astype(int)
df_pool.loc[df_pool['Documents_Complete'] == True, 'Time_Spent_Seconds'] = df_pool.loc[df_pool['Documents_Complete'] == True, 'Time_Spent_Seconds'].apply(lambda x: np.random.normal(loc=350, scale=150)).astype(int)

# Ensure adjusted values are within bounds
df_pool['Number_of_Clicks'] = df_pool['Number_of_Clicks'].apply(lambda x: max(5, min(150, x)))
df_pool['Time_Spent_Seconds'] = df_pool['Time_Spent_Seconds'].apply(lambda x: max(10, min(900, x))) # Max 15 mins

# Calculate a simplified 'Payment_to_Price_Ratio' as a proxy for affordability
df_pool['Payment_to_Price_Ratio'] = df_pool['Monthly_Payment_DT'] * df_pool['Repayment_Duration_Years'] * 12 / df_pool['Total_Acquisition_Price_DT']

# Define approval rules to generate the 'Approved' target variable
def set_approval(row):
    if not row['Documents_Complete']:
        return 1 if np.random.rand() < 0.02 else 0 # Small chance of approval despite incomplete docs

    approved_score = 1.0

    # Influence of Professional Status
    if row['Professional_Status'] == 'Professionnel À Usage Privé':
        approved_score *= 0.7
    elif row['Professional_Status'] == 'Salarié Et Professionnel À Usage Privé / Rentier':
        approved_score *= 0.9

    # Influence of Sector
    if row['Sector'] == 'Secteur Privé':
        approved_score *= 0.9
    elif row['Sector'] == 'Secteur Public':
        approved_score *= 1.1

    # Influence of Existing Loan
    if row['Existing_Loan'] == 'Oui':
        approved_score *= 0.5

    # Influence of Payment Burden (simplified 'Payment_to_Price_Ratio')
    if row['Payment_to_Price_Ratio'] > 0.4:
        approved_score *= 0.3
    elif row['Payment_to_Price_Ratio'] > 0.3:
        approved_score *= 0.6
    elif row['Payment_to_Price_Ratio'] < 0.15:
        approved_score *= 1.3
    elif row['Payment_to_Price_Ratio'] < 0.25:
        approved_score *= 1.1

    # Influence of Repayment Duration
    if row['Repayment_Duration_Years'] > 5:
        approved_score *= 0.8
    elif row['Repayment_Duration_Years'] < 4:
        approved_score *= 1.1

    # NEW: Influence of user interaction metrics
    # Higher clicks and time spent should generally have a positive influence
    if row['Number_of_Clicks'] > 70:
        approved_score *= 1.05 # Slightly better chance for very engaged users
    elif row['Number_of_Clicks'] < 30:
        approved_score *= 0.95 # Slightly lower chance for less engaged users

    if row['Time_Spent_Seconds'] > 450: # More than 7.5 minutes
        approved_score *= 1.05
    elif row['Time_Spent_Seconds'] < 150: # Less than 2.5 minutes
        approved_score *= 0.95

    final_score = approved_score + np.random.uniform(-0.25, 0.25)
    return 1 if final_score > 0.8 else 0

df_pool['Approved'] = df_pool.apply(set_approval, axis=1)

# Drop the temporary calculation column
df_pool = df_pool.drop(columns=['Payment_to_Price_Ratio'])

# --- Balancing the dataset ---
df_approved = df_pool[df_pool['Approved'] == 1]
df_not_approved = df_pool[df_pool['Approved'] == 0]

# Determine the minimum count between the two classes (or use desired_samples_per_class)
count_approved = len(df_approved)
count_not_approved = len(df_not_approved)

n_approved = min(count_approved, desired_samples_per_class)
n_not_approved = min(count_not_approved, desired_samples_per_class)

df_approved_sampled = df_approved.sample(n=n_approved, random_state=42)
df_not_approved_sampled = df_not_approved.sample(n=n_not_approved, random_state=42)

df_balanced = pd.concat([df_approved_sampled, df_not_approved_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced DataFrame to a CSV file
file_name = 'financing_approval_balanced_dataset_Tunisia_with_interactions.csv'
df_balanced.to_csv(file_name, index=False)

print(f"Initial pool generated with {num_samples_pool} rows.")
print(f"Found {count_approved} 'Approved' and {count_not_approved} 'Not Approved' in the pool.")
print(f"Balanced dataset created with {len(df_balanced)} rows ({n_approved} 'Approved', {n_not_approved} 'Not Approved').")
print(f"Balanced demo dataset (with interaction metrics) saved to '{file_name}'")

print("\nFirst 5 rows of the balanced dataset:")
print(df_balanced.head())
print("\nDistribution of 'Approved' status in the BALANCED dataset:")
print(df_balanced['Approved'].value_counts(normalize=True))
print("\nDistribution of 'Documents_Complete' status in the BALANCED dataset:")
print(df_balanced['Documents_Complete'].value_counts(normalize=True))
print("\nDescriptive statistics for new interaction features:")
print(df_balanced[['Number_of_Clicks', 'Time_Spent_Seconds']].describe())