import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.evaluation.single_table import evaluate_quality
import json

# --- Step 1: Data Loading ---
print("Step 1: Script setup and data loading...")
try:
    df = pd.read_parquet('datasets/info_gain_dataset.parquet')
    print("\nSuccessfully loaded file")
except FileNotFoundError:
    print("\n--- ERROR ---")
    print("File not found.")
    exit()
except Exception as e:
    print(f"\nAn error occurred: {e}")
    exit()
print("\nStep 1 Complete.")


# --- Step 2: Define Metadata ---
print("\nStep 2: Defining metadata...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df)

# Print the detected metadata
print("\nDetected Metadata:")
# Print as json for better readability
print(json.dumps(metadata.to_dict(), indent=2))




