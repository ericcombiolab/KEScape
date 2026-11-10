import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import FitnessDataset, collate_variable_length
from model import FitnessPredictor


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = BASE_DIR / 'checkpoint' / 'model_small_bestvalloss_all9_pooled_seed1000.pt'


def parse_args():
    parser = argparse.ArgumentParser(description='Run KEScape inference on a CSV dataset.')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT,
                        help='KEScape checkpoint file path')
    parser.add_argument('--input', type=Path, required=True,
                        help='input CSV with wildtype_sequence and mutant_sequence columns')
    parser.add_argument('--output', type=Path, required=True,
                        help='output CSV path')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='inference batch size (default: 1)')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch_size must be at least 1')
    return args


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        metadata = checkpoint.get('meta') or {}
        state_dict = checkpoint['model_state']
    else:
        metadata = {}
        state_dict = checkpoint

    model = FitnessPredictor(
        metadata.get('num_layers', 1),
        metadata.get('num_heads', 8),
        metadata.get('block_embed_dim', 128),
    )
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def run_inference(input_path, output_path, model, batch_size, device):
    df = pd.read_csv(input_path)
    required_columns = {'wildtype_sequence', 'mutant_sequence'}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError('Input CSV is missing required columns: {}'.format(
            ', '.join(sorted(missing_columns))))

    dataset = FitnessDataset(
        df['wildtype_sequence'].values,
        df['mutant_sequence'].values,
        np.zeros(len(df), dtype=np.float32),
        nprocs=8,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_variable_length,
        pin_memory=device.type == 'cuda',
    )

    predictions = []
    with torch.no_grad():
        for (raw_seq, mut_seq, raw_mask, mut_mask, _,
             raw_mutations, mut_mutations) in loader:
            scores, _, _ = model(
                raw_seq.to(device),
                mut_seq.to(device),
                raw_mask.to(device),
                mut_mask.to(device),
                raw_mutations.to(device),
                mut_mutations.to(device),
            )
            predictions.extend(scores.detach().cpu().numpy().tolist())

    results = df.copy()
    results['prediction'] = predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print('Saved {} predictions to {}'.format(len(predictions), output_path))


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device)
    run_inference(args.input, args.output, model, args.batch_size, device)


if __name__ == '__main__':
    main()
