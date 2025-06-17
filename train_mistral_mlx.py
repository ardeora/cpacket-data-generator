# train_mistral_mlx.py (updated version)
import json
import os
from pathlib import Path
from datasets import Dataset
import pandas as pd
from tqdm import tqdm

def prepare_mlx_dataset():
    """Prepare dataset in MLX-LM compatible format"""
    
    print("Loading original dataset...")
    df = pd.read_parquet('datasets/anova_dataset.parquet')
    
    def row_to_structured_text(row):
        """Convert row to XML-like structured format"""
        return (
            f"<flow>"
            f"<header>max:{row['max_header_bytes']} min:{row['min_header_bytes']} "
            f"mean:{row['mean_header_bytes']:.2f} median:{row['median_header_bytes']:.1f} "
            f"mode:{row['mode_header_bytes']:.1f}</header>"
            f"<fwd>max:{row['fwd_max_header_bytes']} min:{row['fwd_min_header_bytes']} "
            f"mean:{row['fwd_mean_header_bytes']:.2f} std:{row['fwd_std_header_bytes']:.2f} "
            f"median:{row['fwd_median_header_bytes']:.1f} cov:{row['fwd_cov_header_bytes']:.2f} "
            f"variance:{row['fwd_variance_header_bytes']:.2f}</fwd>"
            f"<bwd>std:{row['bwd_std_header_bytes']:.2f} cov:{row['bwd_cov_header_bytes']:.2f} "
            f"variance:{row['bwd_variance_header_bytes']:.2f}</bwd>"
            f"<windows>fwd:{row['fwd_init_win_bytes']} bwd:{row['bwd_init_win_bytes']}</windows>"
            f"<flags>rst:{row['rst_flag_counts']} bwd_rst:{row['bwd_rst_flag_counts']} "
            f"psh_pct:{row['psh_flag_percentage_in_total']:.3f} "
            f"rst_pct:{row['rst_flag_percentage_in_total']:.3f}</flags>"
            f"<iat>mean:{row['bwd_packets_IAT_mean']:.1f} max:{row['bwd_packets_IAT_max']:.1f} "
            f"min:{row['bwd_packets_IAT_min']:.1f} total:{row['bwd_packets_IAT_total']:.1f}</iat>"
            f"<handshake>duration:{row['handshake_duration']:.1f} state:{row['handshake_state']}</handshake>"
            f"<timing>mean_delta:{row['mean_bwd_packets_delta_time']:.2f} "
            f"median_delta:{row['median_bwd_packets_delta_time']:.2f}</timing>"
            f"<class>label:{row['label']} activity:{row['activity']}</class>"
            f"</flow>"
        )
    
    # Create training examples
    print("Creating training examples...")
    train_data = []
    valid_data = []
    
    # Sample to reduce dataset size for faster training (adjust as needed)
    sample_size = min(50000, len(df))  # Start with 50k samples
    df_sample = df.sample(n=sample_size, random_state=42)
    
    # Split into train and validation
    train_size = int(0.9 * len(df_sample))
    train_df = df_sample[:train_size]
    valid_df = df_sample[train_size:]
    
    # Create training examples
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Training data"):
        instruction = f"Generate a network flow record with label '{row['label']}' and activity '{row['activity']}'."
        response = row_to_structured_text(row)
        
        train_data.append({
            "text": f"<s>[INST] {instruction} [/INST] {response}</s>"
        })
    
    # Create validation examples
    for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Validation data"):
        instruction = f"Generate a network flow record with label '{row['label']}' and activity '{row['activity']}'."
        response = row_to_structured_text(row)
        
        valid_data.append({
            "text": f"<s>[INST] {instruction} [/INST] {response}</s>"
        })
    
    # Save as JSONL for MLX-LM
    output_dir = 'mlx_training_data'
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = f'{output_dir}/train.jsonl'
    with open(train_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    valid_file = f'{output_dir}/valid.jsonl'
    with open(valid_file, 'w') as f:
        for item in valid_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Created {len(train_data)} training examples")
    print(f"Created {len(valid_data)} validation examples")
    print(f"Saved to {output_dir}/")
    
    return train_file, valid_file

def create_training_script():
    """Create a shell script for MLX training with correct arguments"""
    
    script_content = """#!/bin/bash

# Fine-tune Mistral 7B with MLX-LM
# Using the correct argument format for the current version

MODEL="mistralai/Mistral-7B-Instruct-v0.2"
DATA_DIR="mlx_training_data"
ADAPTER_PATH="models/mistral-netflow"

# Create output directory
mkdir -p $ADAPTER_PATH

# Training with correct arguments
python -m mlx_lm lora \\
    --model $MODEL \\
    --train \\
    --data $DATA_DIR \\
    --adapter-path $ADAPTER_PATH \\
    --batch-size 2 \\
    --learning-rate 5e-5 \\
    --iters 1000 \\
    --val-batches 25 \\
    --steps-per-report 10 \\
    --steps-per-eval 100 \\
    --save-every 500 \\
    --max-seq-length 1024 \\
    --grad-checkpoint \\
    --seed 42

echo "Training complete! Adapter saved to $ADAPTER_PATH"
"""
    
    with open('train_mistral.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('train_mistral.sh', 0o755)
    print("Created train_mistral.sh script")

def create_config_file():
    """Create a configuration file for more complex setups"""
    
    config = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "train": True,
        "data": "mlx_training_data",
        "adapter_path": "models/mistral-netflow",
        "batch_size": 2,
        "learning_rate": 5e-5,
        "iters": 1000,
        "val_batches": 25,
        "steps_per_report": 10,
        "steps_per_eval": 100,
        "save_every": 500,
        "max_seq_length": 1024,
        "grad_checkpoint": True,
        "seed": 42,
        "fine_tune_type": "lora",
        "optimizer": "adamw"
    }
    
    with open('lora_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Created lora_config.json configuration file")

if __name__ == "__main__":
    # Prepare the dataset
    train_file, val_file = prepare_mlx_dataset()
    
    # Create training script
    create_training_script()
    
    # Create config file
    create_config_file()
    
    print("\nNext steps:")
    print("1. Run the training script: ./train_mistral.sh")
    print("   OR use the config file: python -m mlx_lm lora -c lora_config.json")
    print("2. Monitor GPU/memory usage with: sudo powermetrics --samplers gpu_power")
    print("3. Training progress will be shown every 10 steps")
    print("\nNote: You can adjust 'iters' in the script for longer/shorter training")
    print("      Current setting: 1000 iterations (~1-2 hours on M3 Pro)")