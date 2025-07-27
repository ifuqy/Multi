from tinygrad import Tensor, nn
import numpy as np
from tinygrad import Device
from tinygrad.nn import optim
from sklearn.model_selection import train_test_split
from models.cnn_attention import CNN_Attention
from models.se_resnet10_1D import SE_ResNet10_1D
from models.combined_model import CombinedModel
from classifier import Multi_AI
from utils import save_to_pickle, load_from_pickle
from data_utils import load_config, load_datasets, save_pfd_label
from classifier import CosineAnnealingScheduler
import os
from sklearn.utils import shuffle

def load_data_from_config(config_path):

    config = load_config(config_path)
    data_cfg = config["DATA"]
    save_pickle = data_cfg.get("SAVE_PICKLE", False)

    use_dataloader = "DATALOADER" in data_cfg
    use_dataset = "DATASETS" in data_cfg
    use_test = "TEST" in data_cfg.get("DATALOADER", {}) or ("TEST" in data_cfg.get("DATASETS", {}))

    if use_dataloader:

        paths = data_cfg["DATALOADER"]
        
        files_exist = all(os.path.exists(paths.get(k, "")) for k in ["TRAIN", "TEST"])

        if files_exist:
            print("Loading from dataloader .pkl files...")
            train = load_from_pickle(paths["TRAIN"])
            if use_test:
                test = load_from_pickle(paths["TEST"])
            else:
                test = None
            return train, test

    if use_dataset:
        print("Loading from dataset txt files...")
        if use_test: 
            train_pfds, train_target, test_pfds, test_target = load_datasets(config)
        else:
            train_pfds, train_target = load_datasets(config)

        
        from sklearn.model_selection import train_test_split
        random_state = config["TRAIN"]["RANDOM_STATE"]

        X_train, y_train = shuffle(
            train_pfds, train_target, random_state=random_state
        )
  
        combineModel = CombinedModel()
        features = {
            'profiles': {'phasebins': [64, 96, 128]},
            'intervals': {'intervals': [64, 96, 128]}
        }
        multi_AI = Multi_AI(combineModel, features)

        train_loader = multi_AI.create_Combined_dataloader(X_train, y_train, batch_size=config["TRAIN"]["BATCH_SIZE"])

        if use_test:
            test_loader = multi_AI.create_Combined_dataloader(test_pfds, test_target, batch_size=config["TRAIN"]["BATCH_SIZE"])
        else:
            test_loader = None

        if save_pickle:
            save_path = config["SAVE_PATH"]
            save_to_pickle(train_pfds, os.path.join(save_path, 'train_pfds.pkl'))
            save_to_pickle(train_target, os.path.join(save_path, 'train_target.pkl'))

            save_to_pickle(train_loader, os.path.join(save_path, "train_loader.pkl"))

            save_pfd_label(X_train, y_train,os.path.join(save_path, 'train_labels.txt'))

            if use_test:
                save_to_pickle(test_pfds, os.path.join(save_path, 'test_pfds.pkl'))
                save_to_pickle(test_target, os.path.join(save_path, 'test_target.pkl'))
                save_to_pickle(test_loader,os.path.join(save_path, "test_loader.pkl"))
                save_pfd_label(test_pfds, test_target, os.path.join(save_path, 'test_labels.txt'))

        return train_loader, test_loader

    raise ValueError("No valid DATA config found in yaml.")

def main():
    config_path = "./config/finetune_cfg.yaml"
    config = load_config(config_path)

    device_default = Device.DEFAULT
    if config['DEVICE'] != "Default":
        Device.DEFAULT = config['DEVICE']

    train_dataloader, test_dataloader = load_data_from_config(config_path)


    combineModel = CombinedModel()
    features = {
        'profiles': {'phasebins': [64, 96, 128]},
        'intervals': {'intervals': [64, 96, 128]}
    }

    trained_model = config["TRAIN"]["TRAINED_MODEL_PATH"]
    multi_AI = Multi_AI(combineModel, features, ckpt=trained_model)


    loss_func = lambda x, y: x.cross_entropy(y, label_smoothing=config["LOSS_FUNC"]["LABEL_SMOOTHING"])
    optimizer  = optim.AdamW(nn.state.get_parameters(multi_AI),
                            lr=float(config["OPTIMIZER"]["LR"]),
                            eps=float(config["OPTIMIZER"]["EPS"]),
                            weight_decay=float(config["OPTIMIZER"]["WEIGHT_DECAY"]))
    max_epochs = config["TRAIN"]["EPOCHS"]
    scheduler = CosineAnnealingScheduler(optimizer, max_epochs)
    ckpt_path = config["TRAIN"]["SAVE_MODEL_PATH"]

    multi_AI.set(loss_func, optimizer, scheduler, max_epochs, ckpt_path)

    # Print Configuration Parameters
    print("\n=== Configuration Parameters ===")
    print(f"Training Data Shape: {len(train_dataloader[0][0][0]) * (len(train_dataloader) - 1) + len(train_dataloader[len(train_dataloader)-1][0][0])}")
    print(f"Test data Shape: {len(test_dataloader[0][0][0]) * (len(test_dataloader) - 1) + len(test_dataloader[len(test_dataloader)-1][0][0])}")
    print(f"Save Model Path: {ckpt_path}")

    print("\nHyperparameters:")
    print(f"  Batch Size: {config['TRAIN']['BATCH_SIZE']}")
    print(f"  Epochs: {config['TRAIN']['EPOCHS']}")
    print(f"  Learning Rate: {config['OPTIMIZER']['LR']}")
    print(f"  Weight Decay: {config['OPTIMIZER']['WEIGHT_DECAY']}")
    print(f"  Optimizer: {config['OPTIMIZER']['TYPE']}")
    print(f"  Scheduler: {config['OPTIMIZER']['SCHEDULER']}")
    print(f"  Loss Function: {config['LOSS_FUNC']['TYPE']}")
    print(f"  Label Smoothing: {config['LOSS_FUNC']['LABEL_SMOOTHING']}")
    print(f"  Random Seed: {config['TRAIN']['RANDOM_STATE']}")

    print(f"\nDefault Device: {device_default}")
    print(f"Use Device: {Device.DEFAULT}")
    print("=========================\n")

  
    multi_AI.finetune_dataloader(train_dataloader=train_dataloader,  val_dataloader=None, test_dataloader=test_dataloader)


if __name__ == '__main__':
    main()
