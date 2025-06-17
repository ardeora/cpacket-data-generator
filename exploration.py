import pandas as pd
import numpy as np

# Load the dataset from the parquet file
dataset_path = 'datasets/anova_dataset.parquet'
df = pd.read_parquet(dataset_path)

# Basic information about the dataset
print("Dataset Shape:", df.shape)
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print("\n" + "="*50 + "\n")

# Column information
print("Column Names and Data Types:")
for col in df.columns:
    print(f"{col}: {df[col].dtype}")
print("\n" + "="*50 + "\n")

# Check for unique values in label columns
print("Unique values in 'label' column:", df['label'].unique())
print("Value counts for 'label':")
print(df['label'].value_counts())
print("\n")

print("Unique values in 'activity' column:", df['activity'].unique())
print("Value counts for 'activity':")
print(df['activity'].value_counts())
print("\n" + "="*50 + "\n")

# Check for missing values
print("Missing values per column:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("No missing values found!")
print("\n" + "="*50 + "\n")

# Statistical summary of numerical columns
print("Statistical Summary (first 5 numerical columns):")
numerical_cols = df.select_dtypes(include=[np.number]).columns[:5]
print(df[numerical_cols].describe())