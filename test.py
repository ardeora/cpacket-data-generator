#
# --- FULL PIPELINE (Corrected Version) ---
#

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
# CORRECTED IMPORT: Removed get_visual_evaluation
from sdv.evaluation.single_table import evaluate_quality


# --- Step 1: Data Loading ---
print("Step 1: Script setup and data loading...")
try:
    df = pd.read_parquet('datasets/anova_dataset.parquet')
    print("\nSuccessfully loaded 'datasets/anova_dataset.parquet'.")
except FileNotFoundError:
    print("\n--- ERROR ---")
    print("File 'datasets/anova_dataset.parquet' not found.")
    exit()
except Exception as e:
    print(f"\nAn error occurred: {e}")
    exit()
print("\nStep 1 Complete.")


# --- Step 2: Define and Correct Metadata ---
print("\nStep 2: Defining and correcting metadata...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df)

# Metadata Correction
columns_to_fix_as_numerical = [
    'max_header_bytes', 'min_header_bytes', 'median_header_bytes', 'mode_header_bytes',
    'fwd_max_header_bytes', 'fwd_min_header_bytes', 'fwd_mode_header_bytes',
    'rst_flag_counts', 'bwd_rst_flag_counts'
]
for col_name in columns_to_fix_as_numerical:
    metadata.update_column(column_name=col_name, sdtype='numerical')
metadata.update_column(column_name='handshake_state', sdtype='categorical')
print("\nStep 2 Complete. Metadata has been corrected.")


# --- Step 3: Train the CTGAN Synthesizer ---
print("\nStep 3: Training the model...")
synthesizer = CTGANSynthesizer(metadata, epochs=300, verbose=True)
synthesizer.fit(df)
print("\nStep 3 Complete: Model training is finished!")


# --- Step 4: Generate Synthetic Data ---
print("\nStep 4: Generating new data...")
synthetic_data = synthesizer.sample(num_rows=1000)
print("\nStep 4 Complete: 1000 rows of synthetic data generated.")


# --- Step 5: Save the Trained Model ---
print("\nStep 5: Saving the model to a file...")
synthesizer.save(filepath='new_synthesizer.pkl')
print("\nStep 5 Complete: Model saved to 'new_synthesizer.pkl'.")


# --- Step 6: Evaluate the Quality of Synthetic Data ---
print("\nStep 6: Evaluating data quality...")

# Generate a quality report
quality_report = evaluate_quality(
    real_data=df,
    synthetic_data=synthetic_data,
    metadata=metadata
)

print("\n--- Quality Report ---")
print(f"Overall Quality Score: {quality_report.get_score() * 100:.2f}%")

print("\n--- Column Shape Similarity ---")
print(quality_report.get_details('Column Shapes'))

# Generate visual comparisons
print("\n--- Visual Evaluation ---")
print("Generating comparison plots...")
# CORRECTED VISUALIZATION CALL: Using the new method from the report object
fig = quality_report.get_visualization(property_name='Column Shapes')
fig.show()

print("\nStep 6 Complete: Evaluation finished.")