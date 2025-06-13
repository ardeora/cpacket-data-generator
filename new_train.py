import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

# --- Step 1: Data Loading ---
print("Step 1: Script setup and data loading...")

try:
    df = pd.read_parquet('datasets/info_gain_dataset.parquet')
    print("\nSuccessfully loaded 'datasets/info_gain_dataset.parquet'.")
except FileNotFoundError:
    print("\n--- ERROR ---")
    print("File 'datasets/info_gain_dataset.parquet' not found.")
    exit()
except Exception as e:
    print(f"\nAn error occurred: {e}")
    exit()

print("\nStep 1 Complete.")

# --- Step 2: Define Metadata ---
print("\nStep 2: Defining metadata...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df)


# --- Step 3: Train the CTGAN Synthesizer ---
print("\nStep 3: Training the model...")
synthesizer = CTGANSynthesizer(
    metadata,
    enforce_min_max_values=True,  # Ensure generated values stay within original bounds
    enforce_rounding=False,        # Preserve decimal precision
    epochs=300,                    # Number of training epochs (adjust based on quality)
    batch_size=500,                # Batch size for training
    discriminator_dim=(256, 256),  # Discriminator architecture
    generator_dim=(256, 256),      # Generator architecture
    discriminator_lr=2e-4,         # Learning rate for discriminator
    generator_lr=2e-4,             # Learning rate for generator
    discriminator_steps=1,         # Steps to train discriminator
    log_frequency=True,            # Log training progress
    verbose=True,                  # Show progress
)
synthesizer.fit(df)
print("\nStep 3 Complete: Model training is finished!")

# --- Step 5: Save the Trained Model ---
print("\nStep 5: Saving the model to a file...")
synthesizer.save(filepath='info_gain_synthesizer.pkl')
print("\nStep 5 Complete: Model saved to 'info_gain_synthesizer.pkl'.")

