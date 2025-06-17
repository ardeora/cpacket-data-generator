import pandas as pd
import json
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

# Load the dataset
dataset_path = 'datasets/anova_dataset.parquet'
df = pd.read_parquet(dataset_path)

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

# Create output directory
output_dir = 'llm_training_data'
os.makedirs(output_dir, exist_ok=True)

print(f"Total samples: {len(df):,}")

# Stratified split to maintain class distribution
train_df, val_df = train_test_split(
    df, 
    test_size=0.1, 
    stratify=df['label'], 
    random_state=42
)

print(f"Training samples: {len(train_df):,}")
print(f"Validation samples: {len(val_df):,}")
print("\nClass distribution in training set:")
print(train_df['label'].value_counts(normalize=True))

# Format 1: Simple text files (one flow per line)
print("\nCreating simple text format...")
with open(f'{output_dir}/train_flows.txt', 'w') as f:
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Training"):
        f.write(row_to_structured_text(row) + '\n')

with open(f'{output_dir}/val_flows.txt', 'w') as f:
    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Validation"):
        f.write(row_to_structured_text(row) + '\n')

# Format 2: JSONL for OpenAI fine-tuning
print("\nCreating OpenAI JSONL format...")
def create_openai_format(df, filename):
    with open(filename, 'w') as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=filename):
            # Create prompt-completion pairs
            entry = {
                "messages": [
                    {"role": "system", "content": "You are a network flow data generator. Generate realistic network flow records in XML format."},
                    {"role": "user", "content": f"Generate a network flow record with label '{row['label']}' and activity '{row['activity']}'."},
                    {"role": "assistant", "content": row_to_structured_text(row)}
                ]
            }
            f.write(json.dumps(entry) + '\n')

create_openai_format(train_df, f'{output_dir}/train_openai.jsonl')
create_openai_format(val_df, f'{output_dir}/val_openai.jsonl')

# Format 3: Hugging Face format (conversational)
print("\nCreating Hugging Face format...")
def create_huggingface_format(df, filename):
    conversations = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=filename):
        conversation = {
            "conversations": [
                {"from": "human", "value": f"Generate a network flow record with label '{row['label']}' and activity '{row['activity']}'."},
                {"from": "gpt", "value": row_to_structured_text(row)}
            ]
        }
        conversations.append(conversation)
    
    with open(filename, 'w') as f:
        json.dump(conversations, f, indent=2)

create_huggingface_format(train_df, f'{output_dir}/train_huggingface.json')
create_huggingface_format(val_df, f'{output_dir}/val_huggingface.json')

# Create a small sample file for testing
print("\nCreating sample file for testing...")
sample_df = df.groupby('label').apply(lambda x: x.sample(min(len(x), 100))).reset_index(drop=True)
with open(f'{output_dir}/sample_flows.txt', 'w') as f:
    for _, row in sample_df.iterrows():
        f.write(row_to_structured_text(row) + '\n')

print(f"\nSample file created with {len(sample_df)} flows")

# Save metadata
metadata = {
    "total_samples": len(df),
    "train_samples": len(train_df),
    "val_samples": len(val_df),
    "features": list(df.columns),
    "labels": df['label'].value_counts().to_dict(),
    "activities": df['activity'].value_counts().to_dict()
}

with open(f'{output_dir}/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\nData preparation complete!")
print(f"Files saved in '{output_dir}/' directory")
print("\nNext steps:")
print("- For OpenAI: Use train_openai.jsonl with their fine-tuning API")
print("- For local Llama/Mistral: Use train_flows.txt or train_huggingface.json")
print("- For testing: Use sample_flows.txt to verify format")