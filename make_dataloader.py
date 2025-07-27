import numpy as np
from utils import load_from_pickle, save_to_pickle
import argparse
import glob, os
from models.cnn_attention import CNN_Attention
from models.se_resnet10_1D import SE_ResNet10_1D
from models.combined_model import CombinedModel
from classifier import Multi_AI
from data_utils import save_pfd_label
from sklearn.utils import shuffle

"""
Usage:
    python make_dataloader.py --pfd_dir ./pfds/ --output_root ./test_dataloader --batch_size 256
    python generate_dataloader.py --pfd_list ./mwa_pulsars.txt --output_root ./mwa_test --batch_size 256
"""

def read_pfd_txt_file(filename):
    with open(filename, 'r') as f:
        header = f.readline()
        second_line = f.readline().strip()
        num_columns = len(second_line.split())

    if num_columns == 2:
        dtype = [('fname', '|S200'), ('label', int)]
    elif num_columns == 1:
        dtype = [('fname', '|S200')]
    else:
        raise ValueError(f"Unexpected number of columns: {num_columns}")

    data = np.loadtxt(filename, dtype=dtype, comments='#')
    pfds = data['fname'].astype(str)
    targets = data['label'] if 'label' in data.dtype.names else None
    return pfds, targets

def main():
    parser = argparse.ArgumentParser(description="Generate a dataloader and save it as a pickle file.")

    parser.add_argument('--output_root', type=str, required=True, help='Output prefix path (e.g., ./test_dataloader)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for dataloader')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pfd_dir', type=str, help='Directory containing .pfd files (no labels)')
    group.add_argument('--pfd_list', type=str, help='Text file listing .pfd files and optional labels')

    args = parser.parse_args()

    # Load model
    combineModel = CombinedModel()
    features = {
        'profiles': {'phasebins': [64, 96, 128]},
        'intervals': {'intervals': [64, 96, 128]}
    }
    multi_AI = Multi_AI(combineModel, features)

    if args.pfd_dir:
        pfds = glob.glob(os.path.join(args.pfd_dir, "*.pfd"))
        targets = None
    else:
        pfds, targets = read_pfd_txt_file(args.pfd_list)
        pfds, targets = shuffle(
            pfds, targets, random_state=None
        )

    # Build dataloader
    dataloader = multi_AI.create_Combined_dataloader(pfds, targets, batch_size=args.batch_size)

    # Save outputs
    save_to_pickle(dataloader, args.output_root + '.pkl')
    save_pfd_label(pfds, targets, args.output_root + '.txt')

    print(f"[Done] Saved dataloader to {args.output_root + '.pkl'}")
    print(f"[Done] Saved pfd file list to {args.output_root + '.txt'}")

if __name__ == '__main__':
    main()


