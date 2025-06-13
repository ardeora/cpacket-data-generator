import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.evaluation.single_table import evaluate_quality
import os
import time
import gc
import logging
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# --- GCP-Optimized Configuration ---
# Set environment variables for better performance
os.environ['OMP_NUM_THREADS'] = '4'  # Adjust based on your GCP instance
os.environ['MKL_NUM_THREADS'] = '4'

# --- Logging Setup ---
def setup_logging():
    """Setup logging for GCP Cloud Logging compatibility"""
    log_filename = f'ctgan_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Also log to console for GCP
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# --- Configuration ---
CONFIG = {
    'data_path': 'datasets/info_gain_dataset.parquet',
    'model_save_path': 'info_gain_synthesizer.pkl',
    'checkpoint_dir': 'checkpoints/',
    'sample_size': None,  # Set to integer to train on subset first
    
    # Model hyperparameters
    'model_params': {
        'embedding_dim': 128,          # Reduced from 256 for faster training
        'enforce_min_max_values': True,
        'enforce_rounding': False,
        'epochs': 300,                 # Reduced from 500 - often sufficient
        'batch_size': 1000,            # Increased for better GPU utilization
        'discriminator_dim': (256, 256),
        'generator_dim': (256, 256),
        'discriminator_lr': 2e-4,
        'generator_lr': 2e-4,
        'discriminator_steps': 1,
        'log_frequency': True,
        'verbose': True,
        'cuda': True
    },
    
    # Training optimizations
    'patience': 50,  # Early stopping patience
    'validation_size': 10000,  # Size of validation synthetic data
}

# --- Step 1: Data Loading with Memory Optimization ---
def load_data(config):
    """Load data with error handling and optional sampling"""
    logger.info("Step 1: Loading data...")
    
    try:
        # For large datasets, consider reading in chunks
        df = pd.read_parquet(config['data_path'])
        logger.info(f"Successfully loaded data: {df.shape}")
        
        # Optional: Sample for testing
        if config['sample_size']:
            df = df.sample(n=config['sample_size'], random_state=42)
            logger.info(f"Sampled to {df.shape[0]} rows for testing")
        
        # Memory optimization: convert int64 to int32 where possible
        for col in df.select_dtypes(include=['int64']).columns:
            if df[col].max() < 2**31:
                df[col] = df[col].astype('int32')
        
        logger.info(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

# --- Step 2: Metadata Creation ---
def create_metadata(df):
    """Create and validate metadata"""
    logger.info("Step 2: Creating metadata...")
    
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)
    
    # Log metadata summary
    categorical_cols = sum(1 for info in metadata.columns.values() if info['sdtype'] == 'categorical')
    numerical_cols = sum(1 for info in metadata.columns.values() if info['sdtype'] == 'numerical')
    
    logger.info(f"Metadata created: {categorical_cols} categorical, {numerical_cols} numerical columns")
    
    # Save metadata for reproducibility
    with open('metadata_config.json', 'w') as f:
        json.dump(metadata.to_dict(), f, indent=2)
    
    return metadata

# --- Step 3: Training with Progress Monitoring ---
def train_model(df, metadata, config):
    """Train CTGAN with monitoring and checkpointing"""
    logger.info("Step 3: Starting model training...")
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Initialize synthesizer
    synthesizer = CTGANSynthesizer(metadata, **config['model_params'])
    
    # Training timing
    start_time = time.time()
    
    try:
        # Custom callback for monitoring (if needed)
        synthesizer.fit(df)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time/60:.2f} minutes")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    return synthesizer

# --- Step 4: Model Evaluation ---
def evaluate_model(synthesizer, real_data, config):
    """Evaluate the trained model"""
    logger.info("Step 4: Evaluating model quality...")
    
    # Generate synthetic samples
    synthetic_data = synthesizer.sample(num_rows=config['validation_size'])
    
    # Calculate quality metrics
    quality_report = evaluate_quality(
        real_data.sample(n=min(len(real_data), config['validation_size'])),
        synthetic_data,
        metadata
    )
    
    logger.info(f"Overall Quality Score: {quality_report.get_score():.4f}")
    
    # Save detailed report
    with open('quality_report.json', 'w') as f:
        json.dump(quality_report.get_details('Column Shapes'), f, indent=2)
    
    return quality_report

# --- Step 5: Save Model and Artifacts ---
def save_model(synthesizer, config):
    """Save model and create metadata package"""
    logger.info("Step 5: Saving model and artifacts...")
    
    # Save the model
    synthesizer.save(filepath=config['model_save_path'])
    logger.info(f"Model saved to '{config['model_save_path']}'")
    
    # Save training configuration
    with open('training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create a summary file
    summary = {
        'training_completed': datetime.now().isoformat(),
        'model_path': config['model_save_path'],
        'data_shape': f"{df.shape[0]} rows x {df.shape[1]} columns",
        'config': config
    }
    
    with open('training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # Load data
        df = load_data(CONFIG)
        
        # Create metadata
        metadata = create_metadata(df)
        
        # Train model
        synthesizer = train_model(df, metadata, CONFIG)
        
        # Evaluate model
        quality_report = evaluate_model(synthesizer, df, CONFIG)
        
        # Save everything
        save_model(synthesizer, CONFIG)
        
        # Cleanup
        gc.collect()
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    finally:
        # Ensure logs are flushed
        logging.shutdown()

# --- GCP Deployment Notes ---
"""
To deploy on GCP:

1. Use a GPU-enabled instance (e.g., n1-highmem-8 with Tesla T4/V100)
2. Install CUDA drivers if using GPU
3. Consider using GCS for data storage:
   - gs://your-bucket/datasets/info_gain_dataset.parquet
4. Use Cloud Logging for monitoring
5. Set up Cloud Scheduler for periodic retraining

Example startup script:
#!/bin/bash
pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES=0
python train_ctgan_gcp.py
"""