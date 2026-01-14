import numpy as np
from models.cnn_attention import CNN_Attention
from models.se_resnet10_1D import SE_ResNet10_1D
from models.combined_model import CombinedModel
from tinygrad import Device
from classifier import Multi_AI
from data_utils import read_pfd_label
import glob
import argparse
import os, gc
from utils import load_from_pickle

"""
Usage:
python predict.py --ckpt ./trained_model/weight_0.9954_0.9830.pth --outfile clf_result.txt --pfd_dataloader test_dataloader.pkl --pfd_file ./test_pfdfile.txt --use_prob
python predict.py --ckpt ./trained_model/weight_0.9954_0.9830.pth --outfile ./test_pfdfile.txt --pfd_dir ./test_pfds/ --use_prob --chunk_size 500
"""

def main():
    parser = argparse.ArgumentParser(description="Predict the input pulsar candidates (.pfd) or (.json)")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--outfile', type=str, required=True, help='Output file to save predictions')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pfd_dir', type=str, help='Directory containing .pfd or .json files (for direct predict)')
    parser.add_argument('--chunk_size', type=int, default=500, help='Max number of pfds loaded each time (Reduce if memory is insufficient.)')

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
        pfds += glob.glob(os.path.join(args.pfd_dir, "*.json"))

        pfds = [p for p in pfds if p and os.path.isfile(p)]
        pfds.sort()

        total = len(pfds)
        chunk_size = args.chunk_size
        num_chunks = (total + chunk_size - 1) // chunk_size
        print(f"Total files: {total}, Chunk size: {chunk_size}, Total chunks: {num_chunks}")

        with open(args.outfile, "w", encoding="utf-8") as fw:
            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, total)

                pfds_chunk = pfds[start:end]
                print(f"\n[Chunk {i+1}/{num_chunks}] Predicting {len(pfds_chunk)} files...")

                if args.use_prob:
                    results = multi_AI.predict_prob(pfds_chunk)
                else:
                    results = multi_AI.predict(pfds_chunk)

                for pfd, res in zip(pfds_chunk, results):
                    fw.write(f"{pfd}\t{res}\n")

                # Free memory
                del pfds_chunk, results
                gc.collect()

    print(f"\n Prediction Done. Output saved to: {args.outfile}")

if __name__ == '__main__':
    main()