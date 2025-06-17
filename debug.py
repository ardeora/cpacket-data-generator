import json
import pandas as pd
import re
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, generate
# Import the make_sampler function
from mlx_lm.sample_utils import make_sampler
from tqdm import tqdm
import argparse

class SyntheticFlowGenerator:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2", adapter_path="models/mistral-netflow"):
        """Initialize the generator with the base model and LoRA adapter"""
        print(f"Loading base model {model_id} with adapter {adapter_path}...")
        self.model, self.tokenizer = load(model_id, adapter_path=adapter_path, lazy=False)
        print("Model loaded successfully!")

    def generate_flow(self, label=None, activity=None, temp=0.7, max_tokens=512):
        """Generate a single network flow record"""
        if activity:
            prompt = f"<s>[INST] Generate a network flow record with label '{label}' and activity '{activity}'. [/INST]"
        elif label:
            prompt = f"<s>[INST] Generate a network flow record with label '{label}'. [/INST]"
        else:
            prompt = "<s>[INST] Generate a network flow record. [/INST]"
            
        
        # Create a sampler with the specified temperature
        sampler = make_sampler(temp=temp)
        
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            sampler=sampler,
            max_tokens=max_tokens,
            verbose=False
        )

        print(f"Generated response: {response}")

        flow_match = re.search(r'<flow>.*?</flow>', response, re.DOTALL)
        if flow_match:
            return flow_match.group(0)
        return None

    def parse_flow_to_dict(self, flow_text):
        try:
            patterns = {
                'max_header_bytes': r'<header>max:(\d+)',
                'min_header_bytes': r'<header>.*?min:(\d+)',
                'mean_header_bytes': r'<header>.*?mean:([\d.]+)',
                'median_header_bytes': r'<header>.*?median:([\d.]+)',
                'mode_header_bytes': r'<header>.*?mode:([\d.]+)',
                'fwd_max_header_bytes': r'<fwd>max:(\d+)',
                'fwd_min_header_bytes': r'<fwd>.*?min:(\d+)',
                'fwd_mean_header_bytes': r'<fwd>.*?mean:([\d.]+)',
                'fwd_std_header_bytes': r'<fwd>.*?std:([\d.]+)',
                'fwd_init_win_bytes': r'<windows>fwd:(\d+)',
                'bwd_init_win_bytes': r'<windows>.*?bwd:(\d+)',
                'rst_flag_counts': r'<flags>rst:(\d+)',
                'handshake_state': r'<handshake>.*?state:(\d+)',
                'label': r'<class>label:(\w+)',
                'activity': r'<class>.*?activity:([\w-]+)'
            }

            flow_dict = {}
            for key, pattern in patterns.items():
                match = re.search(pattern, flow_text)
                if match:
                    value = match.group(1)
                    if key in ['label', 'activity']:
                        flow_dict[key] = value
                    elif '.' in value:
                        flow_dict[key] = float(value)
                    else:
                        flow_dict[key] = int(value)

            return flow_dict
        except Exception as e:
            print(f"Error parsing flow: {e}")
            return None

    def generate_dataset(self):
        activity_label = {
          'Benign': 'Benign',
          'Benign-Systemic': 'Benign',
          'Benign-Telnet': 'Benign',
          'Benign-Web_Browsing_HTTP-S': 'Benign',
          'Benign-SSH': 'Benign',
          'Benign-Email-Send': 'Benign',
          'Benign-Email-Receive': 'Benign',
          'Benign-FTP': 'Benign',
          'Attack-TCP-BYPass-V1': 'Attack',
          'Attack-TCP-Flag-SYN': 'Attack',
          'Attack-TCP-Flag-ACK': 'Attack',
          'Attack-TCP-Flag-ACK-PSH': 'Attack',
          'Attack-Killer-TCP': 'Attack',
          'Attack-TCP-Valid-SYN': 'Attack',
          'Attack-TCP-Flag-SYN-ACK': 'Attack',
          'Attack-TCP-IGMP': 'Attack',
          'Attack-TCP-Flag-MIX': 'Attack',
          'Attack-Killall-v2': 'Attack',
          'Attack-TCP-SYN': 'Attack',
          'Attack-TCP-Control': 'Attack',
          'Attack-TCP-Flag-SYN-TIME': 'Attack',
          'Attack-TCP-Flag-SYN-TFO': 'Attack',
          'Attack-TCP-Flag-OSYNP': 'Attack',
          'Attack-TCP-Flag-OSYN': 'Attack',
          'Attack-TCP-Flag-RST-ACK': 'Attack',
          'Suspicious': 'Suspicious'
        }
        
        synthetic_flows = []
        print("Generating synthetic flows...")
        # Create each flow with a label and activity
        for activity, label in activity_label.items():
            print(f"\nGenerating flows for label '{label}' with activity '{activity}'...")
            for _ in tqdm(range(1)):
                flow_text = self.generate_flow(label=label, activity=activity, temp=0.8)
                if flow_text:
                    flow_dict = self.parse_flow_to_dict(flow_text)
                    if flow_dict:
                        synthetic_flows.append(flow_dict)

        df = pd.DataFrame(synthetic_flows)
        original_columns = ['max_header_bytes', 'min_header_bytes', 'mean_header_bytes',
                            'median_header_bytes', 'mode_header_bytes', 'fwd_max_header_bytes',
                            'fwd_min_header_bytes', 'fwd_mean_header_bytes', 'fwd_std_header_bytes',
                            'fwd_median_header_bytes', 'fwd_cov_header_bytes', 'fwd_mode_header_bytes',
                            'fwd_variance_header_bytes', 'bwd_std_header_bytes', 'bwd_cov_header_bytes',
                            'bwd_variance_header_bytes', 'fwd_init_win_bytes', 'bwd_init_win_bytes',
                            'rst_flag_counts', 'bwd_rst_flag_counts', 'psh_flag_percentage_in_total',
                            'rst_flag_percentage_in_total', 'fwd_psh_flag_percentage_in_total',
                            'fwd_syn_flag_percentage_in_total', 'bwd_psh_flag_percentage_in_total',
                            'fwd_psh_flag_percentage_in_fwd_packets', 'bwd_psh_flag_percentage_in_bwd_packets',
                            'bwd_rst_flag_percentage_in_bwd_packets', 'bwd_packets_IAT_mean',
                            'bwd_packets_IAT_max', 'bwd_packets_IAT_min', 'bwd_packets_IAT_total',
                            'bwd_packets_IAT_median', 'bwd_packets_IAT_mode', 'handshake_duration',
                            'handshake_state', 'mean_bwd_packets_delta_time', 'median_bwd_packets_delta_time',
                            'skewness_packets_delta_len', 'mode_fwd_packets_delta_len', 'label', 'activity']

        for col in original_columns:
            if col not in df.columns:
                df[col] = 'Unknown' if col in ['label', 'activity'] else 0.0

        return df[original_columns]

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic network flow data')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='synthetic_flows.parquet', help='Output file path')
    parser.add_argument('--model-id', type=str, default='mistralai/Mistral-7B-Instruct-v0.2', help='Base model ID or path')
    parser.add_argument('--adapter-path', type=str, default='models/mistral-netflow', help='Path to LoRA adapter directory')
    args = parser.parse_args()

    generator = SyntheticFlowGenerator(model_id=args.model_id, adapter_path=args.adapter_path)
    synthetic_df = generator.generate_dataset()
    synthetic_df.to_csv(args.output, index=False)

    print(f"\nSynthetic dataset saved to {args.output}")
    print(f"Shape: {synthetic_df.shape}")
    print("\nLabel distribution:")
    print(synthetic_df['label'].value_counts())

if __name__ == "__main__":
    main()