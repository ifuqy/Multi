import numpy as np
import yaml
from data_loader import pfdreader, dataloader
import os, glob
import argparse
from models.cnn_attention import CNN_Attention
from models.se_resnet10_1D import SE_ResNet10_1D
from models.combined_model import CombinedModel
from classifier import Multi_AI

def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config

def load_datasets(config):
    if len(config['DATA']['DATASETS']) == 1:
        all_pfds = []
        all_targets = []
        data_config = config['DATA']['DATASETS']['TRAIN']
        for dataset_name, categories in data_config.items():
            for category, path in categories.items():
                try:
                    loader = dataloader(path)
                    all_pfds.extend(loader.pfds)
                    all_targets.extend(loader.target)
                    print(f"Loaded {dataset_name} {category}: {len(loader.pfds)} samples")
                except Exception as e:
                    print(f"Error loading {dataset_name}/{category}: {str(e)}")
        
        return np.array(all_pfds), np.array(all_targets)
    
    elif len(config['DATA']['DATASETS']) == 2:
        train_pfds = []
        train_targets = []
        test_pfds = []
        test_targets = []
        
        train_config = config['DATA']['DATASETS']['TRAIN']
        for dataset_name, categories in train_config.items():
            for category, path in categories.items():                
                try:
                    loader = dataloader(path)
                    train_pfds.extend(loader.pfds)
                    train_targets.extend(loader.target)
                    print(f"Loaded TRAIN/{dataset_name}/{category}: {len(loader.pfds)} samples")
                except Exception as e:
                    print(f"Error loading TRAIN/{dataset_name}/{category}: {str(e)}")
        
        val_config = config['DATA']['DATASETS']['TEST']
        for dataset_name, categories in val_config.items():
            for category, path in categories.items():
                try:
                    loader = dataloader(path)
                    test_pfds.extend(loader.pfds)
                    test_targets.extend(loader.target)
                    print(f"Loaded TEST/{dataset_name}/{category}: {len(loader.pfds)} samples")
                except Exception as e:
                    print(f"Error loading TEST/{dataset_name}/{category}: {str(e)}")
        
        return (
            np.array(train_pfds), 
            np.array(train_targets),
            np.array(test_pfds),
            np.array(test_targets)
        )
    
def save_pfd_label(X_data, y_data=None, output_file='labels.txt'):
    """
    Save a list of .pfd filenames and optional labels to a text file.

    Args:
        X_data (List): List of data objects, each with a 'pfd_filename' attribute.
        y_data (List or None): Optional list of labels. If None, only filenames will be saved.
        output_file (str): Path to the output file.
    """
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            if y_data is not None:
                f.write("Filename\tLabel\n")
                for x, y in zip(X_data, y_data):
                    filename = getattr(x, 'pfd_filename', '')
                    f.write(f"{filename}\t{y}\n")
                print(f"Successfully saved {len(X_data)} records with labels to file: {output_file}")
            else:
                f.write("Filename\n")
                for x in X_data:
                    filename = getattr(x, 'pfd_filename', '')
                    f.write(f"{filename}\n")
                print(f"Successfully saved {len(X_data)} filenames to file: {output_file}")
                
    except Exception as e:
        print(f"Error saving file: {str(e)}")

def read_pfd_label(input_file='labels.txt'):
    """
    Read .pfd filenames and optional labels from a text file.

    Args:
        input_file (str): Path to the input file.

    Returns:
        Tuple[List[str], List[str] or None]: A tuple (filenames, labels).
            If no labels are present, labels will be None.
    """
    try:
        filenames = []
        labels = []

        with open(input_file, 'r', encoding='utf-8') as f:
            header = next(f)  # Skip header line
            has_label = 'Label' in header

            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if has_label and len(parts) >= 2:
                    filenames.append(parts[0])
                    labels.append(parts[1])
                elif not has_label and len(parts) >= 1:
                    filenames.append(parts[0])
                else:
                    print(f"Warning: Incorrectly formatted line '{line}'")

        print(f"Successfully read {len(filenames)} records from file: {input_file}")
        return filenames, labels if has_label else None

    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return [], None
    
def result_to_file(X_pfd, y_label, val_preds_prob, output_file):
    try:
        with open(output_file, 'w') as file:
            # Write header
            file.write("X_pfd\ty_label\tval_preds_prob\n")
            
            # Iterate over all samples
            for x, y, prob in zip(X_pfd, y_label, val_preds_prob):
                # Join each feature of X_pfd with commas
                x_str = ','.join(map(str, x)) if isinstance(x, (list, np.ndarray)) else str(x)
                # Write a line with each part separated by tabs
                file.write(f"{x_str}\t{y}\t{prob}\n")
                
        print(f"Data has been successfully written to file: {output_file}")
    except Exception as e:
        print(f"Error writing to file: {e}")

def generate_pfd_list(source_dir, output_file, prefix, label):
    """
    Generate a text file containing paths of .pfd files with specified prefix and label.
    
    Args:
        source_dir (str): Directory path containing .pfd files.
        output_file (str): Output text file path.
        prefix (str): Prefix to add before each .pfd file path.
        label (int): Label number to append after each .pfd file path.
    """
    # Get all .pfd files in the directory
    pfd_files = [f for f in os.listdir(source_dir) if f.endswith('.pfd')]
    
    # Construct lines to write
    lines = [f"{prefix}{filename} {label}" for filename in pfd_files]
    
    # Write to output file
    try:
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        print(f"Successfully generated: {output_file}")
        print(f"Processed {len(lines)} files")
    except Exception as e:
        print(f"Error writing file: {e}")


if __name__ == "__main__":
    """
    # Configuration 
    SOURCE_DIR = "/data_zfs/fast/fuqy/AI/data/Multi_AI/Parkes/pulsars"
    OUTPUT_FILE = "/data_zfs/fast/fuqy/AI/data/Multi_AI/Parkes/parkes_pulsars.txt"
    PATH_PREFIX = "../data/Multi_AI/Parkes/pulsars/"
    LABEL_NUMBER = 1
    
    # Generate the .pfd list file
    generate_pfd_list(SOURCE_DIR, OUTPUT_FILE, PATH_PREFIX, LABEL_NUMBER)
    """

