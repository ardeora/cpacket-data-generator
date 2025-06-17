#
# --- SCRIPT TO GENERATE SYNTHETIC DATA FROM A SAVED MODEL ---
#

import pandas as pd
from sdv.single_table import CTGANSynthesizer
import os

print("--- Starting Data Generation Script ---")

# --- 1. Define File Paths ---
# The path to your saved, trained model.
# Note: Your last log output said the file was 'new_synthesizer.pkl'.
# If you used a different name, change it here.
model_filepath = 'info_gain_synthesizer.pkl' 

# The name for the output CSV file.
output_csv_path = 'synthetic_network_data_1000_rows.csv'


# --- 2. Load the Trained Synthesizer ---
print(f"Loading the trained model from '{model_filepath}'...")

try:
    # Check if the model file exists before trying to load it
    if not os.path.exists(model_filepath):
        raise FileNotFoundError(f"Error: The model file '{model_filepath}' was not found in this directory.")
        
    synthesizer = CTGANSynthesizer.load(filepath=model_filepath)
    print("Model loaded successfully!")

except Exception as e:
    print(e)
    # Exit the script if the model can't be loaded
    exit()


# --- 3. Generate New Synthetic Data ---
num_rows_to_generate = 1000  # Change this number to generate more or fewer rows
print(f"Generating {num_rows_to_generate} new rows of synthetic data...")

synthetic_data = synthesizer.sample(num_rows=num_rows_to_generate)
print("Data generation complete.")


# --- 4. Save the Data to a CSV File ---
print(f"Saving the synthetic data to '{output_csv_path}'...")

# We use index=False to prevent pandas from writing a new row index into the CSV file.
synthetic_data.to_csv(output_csv_path, index=False)
print("File saved successfully.")

print("\n--- Script Finished ---")
print(f"You can now find your {num_rows_to_generate} new data samples in the file: {output_csv_path}")