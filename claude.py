import pandas as pd
from sdv.metadata import SingleTableMetadata
import json

# --- Step 1: Data Loading ---
print("Step 1: Script setup and data loading...")
df = pd.read_parquet('datasets/info_gain_dataset.parquet')
print(f"Dataset shape: {df.shape}")

# --- Step 2: Define Metadata ---
print("\nStep 2: Defining metadata...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df)

# --- Step 3: Analyze Column Types ---
print("\n" + "="*60)
print("COLUMN TYPE ANALYSIS")
print("="*60)

# Get columns by type
categorical_columns = []
numerical_columns = []

for column, info in metadata.columns.items():
    if info['sdtype'] == 'categorical':
        categorical_columns.append(column)
    elif info['sdtype'] == 'numerical':
        numerical_columns.append(column)

print(f"\nTotal Categorical Columns: {len(categorical_columns)}")
print(f"Total Numerical Columns: {len(numerical_columns)}")

print("\n--- Categorical Columns ---")
for col in categorical_columns:
    nunique = df[col].nunique()
    dtype = df[col].dtype
    print(f"{col:35} | Unique values: {nunique:6} | dtype: {dtype}")

print("\n--- Numerical Columns ---")
for col in numerical_columns[:10]:  # Show first 10
    dtype = df[col].dtype
    print(f"{col:35} | dtype: {dtype}")
print(f"... and {len(numerical_columns)-10} more numerical columns")

# --- Step 4: Investigate Categorical Columns ---
print("\n" + "="*60)
print("INVESTIGATING CATEGORICAL COLUMNS")
print("="*60)

print("\nChecking why some 'bytes' columns are categorical:")
for col in categorical_columns:
    if 'bytes' in col.lower() and col not in ['label', 'activity']:
        print(f"\n{col}:")
        print(f"  Data type: {df[col].dtype}")
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Value counts:\n{df[col].value_counts().head()}")

# --- Step 5: Manual Override Example ---
print("\n" + "="*60)
print("MANUAL METADATA OVERRIDE EXAMPLE")
print("="*60)

# If you want to manually override column types
print("\nTo manually override column types, you can do:")
print("""
# Example: Change 'max_header_bytes' from categorical to numerical
metadata.update_column(
    column_name='max_header_bytes',
    sdtype='numerical'
)

# Or update multiple columns at once
columns_to_fix = ['max_header_bytes', 'min_header_bytes', 'fwd_max_header_bytes', 
                  'fwd_min_header_bytes', 'median_header_bytes', 'mode_header_bytes',
                  'fwd_mode_header_bytes']

for col in columns_to_fix:
    metadata.update_column(column_name=col, sdtype='numerical')
""")

# --- Step 6: Data Type Summary ---
print("\n" + "="*60)
print("DATA TYPE SUMMARY")
print("="*60)

# Create a summary DataFrame
type_summary = pd.DataFrame({
    'Column': list(metadata.columns.keys()),
    'SDV_Type': [info['sdtype'] for info in metadata.columns.values()],
    'Pandas_dtype': [str(df[col].dtype) for col in metadata.columns.keys()],
    'Unique_Values': [df[col].nunique() for col in metadata.columns.keys()]
})

# Show columns where SDV type might not match expected type
print("\nColumns where SDV detected as categorical but might be numerical:")
suspicious = type_summary[
    (type_summary['SDV_Type'] == 'categorical') & 
    (type_summary['Column'].str.contains('bytes|header', case=False)) &
    (~type_summary['Column'].isin(['label', 'activity']))
]
print(suspicious.to_string(index=False))

# --- Step 7: Export Functions ---
print("\n" + "="*60)
print("HELPER FUNCTIONS")
print("="*60)

print("""
# Function to get columns by type
def get_columns_by_type(metadata):
    categorical = []
    numerical = []
    for column, info in metadata.columns.items():
        if info['sdtype'] == 'categorical':
            categorical.append(column)
        elif info['sdtype'] == 'numerical':
            numerical.append(column)
    return categorical, numerical

# Function to validate metadata
def validate_metadata(df, metadata):
    issues = []
    for col, info in metadata.columns.items():
        if info['sdtype'] == 'categorical':
            if df[col].nunique() > 100:
                issues.append(f"{col}: Has {df[col].nunique()} unique values but marked as categorical")
        elif info['sdtype'] == 'numerical':
            if df[col].nunique() < 10:
                issues.append(f"{col}: Has only {df[col].nunique()} unique values but marked as numerical")
    return issues

# Use the functions
categorical_cols, numerical_cols = get_columns_by_type(metadata)
issues = validate_metadata(df, metadata)
if issues:
    print("\\nPotential metadata issues found:")
    for issue in issues:
        print(f"  - {issue}")
""")