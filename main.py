import pandas as pd
import json
import numpy as np

# Load the dataset
dataset_path = 'datasets/anova_dataset.parquet'
df = pd.read_parquet(dataset_path)

def row_to_structured_text(row):
    """Optimized format for LLM training - preserves structure and precision"""
    # Group related features for better pattern learning
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

# Fixed JSON format with proper type conversion
def row_to_json_text(row):
    """JSON-inspired format - easy to parse back"""
    flow = {
        "hdr": {"max": int(row['max_header_bytes']), "min": int(row['min_header_bytes']), 
                "mean": float(round(row['mean_header_bytes'], 2))},
        "fwd": {"max": int(row['fwd_max_header_bytes']), 
                "mean": float(round(row['fwd_mean_header_bytes'], 2)),
                "std": float(round(row['fwd_std_header_bytes'], 2))},
        "win": {"fwd": int(row['fwd_init_win_bytes']), "bwd": int(row['bwd_init_win_bytes'])},
        "flags": {"rst": int(row['rst_flag_counts']), 
                  "rst_pct": float(round(row['rst_flag_percentage_in_total'], 3))},
        "iat": {"mean": float(round(row['bwd_packets_IAT_mean'], 1))},
        "hs": {"dur": float(round(row['handshake_duration'], 1)), 
               "state": int(row['handshake_state'])},
        "cls": {"lbl": str(row['label']), "act": str(row['activity'])}
    }
    return "NETFLOW:" + json.dumps(flow, separators=(',', ':'))

# Test both formats
first_row = df.iloc[0]
print("Structured XML-like format (RECOMMENDED):")
print(row_to_structured_text(first_row))
print(f"Length: {len(row_to_structured_text(first_row))} chars")
print("\n" + "="*80 + "\n")

print("JSON-like format:")
print(row_to_json_text(first_row))
print(f"Length: {len(row_to_json_text(first_row))} chars")
print("\n" + "="*80 + "\n")

# Show examples of each class
print("Examples by class (structured format):")
for label in ['Benign', 'Attack', 'Suspicious']:
    sample = df[df['label'] == label].iloc[0]
    print(f"\n{label}:")
    print(row_to_structured_text(sample))

# Let's also check a few Attack samples to see pattern variety
print("\n" + "="*80 + "\n")
print("Different attack types (structured format):")
attack_types = df[df['label'] == 'Attack']['activity'].unique()[:3]
for attack_type in attack_types:
    sample = df[df['activity'] == attack_type].iloc[0]
    print(f"\n{attack_type}:")
    print(row_to_structured_text(sample))