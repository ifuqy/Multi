import numpy as np
from models.cnn_attention import CNN_Attention
from models.se_resnet10_1D import SE_ResNet10_1D
from models.combined_model import CombinedModel
from tinygrad import Device
from classifier import Multi_AI
from data_utils import read_pfd_label
import glob
import argparse
import os 
from utils import load_from_pickle

"""
Usage:
python predict.py --ckpt ./trained_model/weight_0.9954_0.9830.pth --outfile clf_result.txt --pfd_dataloader test_dataloader.pkl --pfd_file ./test_pfdfile.txt --use_prob
python predict.py --ckpt ./trained_model/weight_0.9954_0.9830.pth --outfile ./test_pfdfile.txt --pfd_dir ./test_pfds/
"""

def main():
    parser = argparse.ArgumentParser(description="Predict the input pulsar candidates (.pfd)")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--outfile', type=str, required=True, help='Output file to save predictions')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pfd_dir', type=str, help='Directory containing .pfd files (for direct predict)')
    group.add_argument('--pfd_dataloader', type=str, help='Specified dataloader file')

    parser.add_argument('--pfd_file', type=str, help='Path to save pfd file list (only required if using dataloader)')
    parser.add_argument('--use_prob', action='store_true', help='Save probability instead of predicted label')
    
    args = parser.parse_args()

    # Initialize model
    combineModel = CombinedModel()
    features = {
        'profiles': {'phasebins': [64, 96, 128]},
        'intervals': {'intervals': [64, 96, 128]}
    }
    multi_AI = Multi_AI(combineModel, features, ckpt=args.ckpt)

    # Load input
    if args.pfd_dataloader:
        if not args.pfd_file:
            raise ValueError("pfd_file is required when using dataloader")
        pfds, _ = read_pfd_label(args.pfd_file)
        pfd_dataloader = load_from_pickle(args.pfd_dataloader)

        if args.use_prob:
            results = multi_AI.classifier.predict_prob(pfd_dataloader)
        else:
            results = multi_AI.classifier.predict(pfd_dataloader)

    else:
        pfds = glob.glob(os.path.join(args.pfd_dir, "*.pfd"))
        
        if args.use_prob:
            results = multi_AI.predict_prob(pfds)
        else:
            results = multi_AI.predict(pfds)

    # Output
    with open(args.outfile, 'w', encoding='utf-8') as f:
        for pfd, result in zip(pfds, results):
            f.write(f"{pfd}\t{result}\n")

    print(f"Successfully wrote {len(results)} results to {args.outfile}")

if __name__ == '__main__':
    main()