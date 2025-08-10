import numpy as np
from multiprocessing import Pool, cpu_count
from tinygrad import Tensor
from tinygrad.nn.state import safe_load, load_state_dict
from typing import List, Optional, Generator, Tuple, Union
from tqdm import tqdm
from utils import normalize
from scipy.interpolate import interp1d
from tinygrad import Device
import yaml

def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config

global_config = load_config('./config/global_cfg.yaml')

Device.DEFAULT = global_config['DEVICE']

use_multiprocessing = global_config['MULTIPROCESSING']['ENABLED']

if use_multiprocessing:
    num_workers = global_config['MULTIPROCESSING']['NUM_WORKERS']

# Load denoise model
class denoise_UNet:
    def __init__(self):
        from models.Basic_UNet import BasicUNet  
        self.model = BasicUNet()

    def load_weights(self, path: str):
        state = safe_load(path)
        load_state_dict(self.model, state)

    def __call__(self, x: Tensor) -> Tensor:
        return self.model(x)

# Process single pfd
def process_single_pfd(pfd, key, values, feature):
    if key in ['phasebins', 'DMbins']:
        if not isinstance(values, list):
            return np.array(pfd.getdata(**feature)).reshape(-1, values)
        else:
            return [np.array(singledata).reshape(-1, values[i]) for i, singledata in enumerate(pfd.getdata(**{key:values}))]
    elif key in ['intervals', 'subbands']:
        if not isinstance(values, list):
            return np.array(pfd.getdata(**feature)).reshape(-1, values, values)
        else:
            return [np.array(singledata).reshape(-1, values[i], values[i]) for i, singledata in enumerate(pfd.getdata(**{key:values}))]

def process_wrapper(args):
    return process_single_pfd(*args)

# Parallel processing
def process_data_in_parallel(pfds, key, values, feature):

    with Pool(processes=num_workers) as pool:

        task_args = [(pfd, key, values, feature) for pfd in pfds]

        results = []
        with Pool(processes=num_workers) as pool:
            for res in tqdm(pool.imap(process_wrapper, task_args), total=len(task_args), 
                            desc=f"Create feature {key} Dataloaders", mininterval=1.0):
                results.append(res)

    if isinstance(values, list):
        return [np.stack([r[i] for r in results]).astype(np.float32) for i in range(len(values))]
    else:
        return np.stack(results).astype(np.float32)

def sim_intervals_harmonic(intervals):
    # intervals.shape=(n,1,64,64)
    x_old = np.arange(intervals.shape[2])

    x_new = np.linspace(0, intervals.shape[2]-1, intervals.shape[2]*2)

    f = interp1d(x_old, intervals, axis=2, kind="cubic")

    intervals = f(x_new)

    odds_front = intervals[:, :, 1::2, ::2] 
    evens_back = intervals[:, :, 0::2, ::2]

    intervals_harmonic = np.concatenate((odds_front, evens_back), axis=3)
    del odds_front, evens_back, x_old, x_new
    
    return normalize(intervals_harmonic)

def sim_profiles_harmonic(profiles):
    # profiles.shape=(n,1,64)
    extended_profiles = np.concatenate((profiles, profiles), axis=2)

    x_old = np.arange(extended_profiles.shape[2])
    x_new = np.linspace(0, extended_profiles.shape[2] - 1, profiles.shape[2])
    f = interp1d(x_old, extended_profiles, axis=2, kind='cubic')

    extended_profiles = f(x_new)
    del x_old, x_new

    return normalize(extended_profiles)


def pfd_to_data(pfds, target=None, feature=None, denoise=True, sim_harmonic=False):
    targetmap = {'phasebins': 1, 'DMbins': 2, 'intervals': 3, 'subbands': 4}
    key = list(feature.keys())[0]
    values = list(feature.values())[0]

    data = process_data_in_parallel(pfds, key, values, feature)

    if denoise and key in ['intervals', 'subbands']:

        model = denoise_UNet()
        model.load_weights('./trained_model/denoise_model.pth')

        if not isinstance(values, list):
            batch_size = 128
            preds = []

            for i in range(0, len(data), batch_size):
                batch = Tensor(data[i:i+batch_size].astype(np.float32))
                preds.append(model(batch).numpy())
                del batch

            data = np.concatenate(preds)
        else:
            for f, sfd in enumerate(data):
                batch_size = 128
                preds = []
                # preds = None
                for i in range(0, len(sfd), batch_size):
                    batch = Tensor(sfd[i:i+batch_size].astype(np.float32))
                    # pred = model(batch)
                    preds.append(model(batch).numpy())
                    # MemoryError: rm_alloc returned 81: Out of memory
                    # if preds is None:
                    #     preds = model(batch).detach()
                    # else:
                    #     preds = preds.cat(model(batch).detach(), dim=0)
                    del batch
                # preds = preds.numpy()
                data[f] = np.concatenate(preds)
        del preds


    if target is not None:
        if sim_harmonic:
            if key in ['intervals']:
                for i, singledata in enumerate(data):
                    data[i] = np.concatenate((data[i], sim_intervals_harmonic(singledata)), axis=0)
            if key in ['phasebins']:
                for i, singledata in enumerate(data):
                    data[i] = np.concatenate((data[i], sim_profiles_harmonic(singledata)), axis=0)    

            target = np.concatenate((target, target), axis=0) 

        target = np.array(target).astype(np.int16) 
                
        if target.ndim == 1:
            mytarget = target
        else:
            mytarget = target[..., targetmap[key]]

        if isinstance(values, list):
            return *data, mytarget
        else:
            return np.array(data), mytarget
    else:
        if isinstance(values, list):
            # return *data,
            data = [np.array(d)[np.newaxis, :, :] for d in data]
            return data 
        else:
            # return np.array(data).astype(np.float32)   
            data = np.array(data)[np.newaxis, :, :]
            return data.astype(np.float32)



class pfd_Dataset:
    def __init__(self, data: np.ndarray, labels: Optional[np.ndarray] = None):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]

def pfd_Dataloader(
    dataset: pfd_Dataset,
    batch_size: int = 64,
    shuffle: bool = True
) -> Generator[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], None, None]:

    if dataset.labels is None:
        indices = np.arange(len(dataset[0]))
        # dataset.data.shape = (1, 1000, 1, 64, 64), so len(dataset[0])
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(dataset[0]), batch_size):
            batch_idx = indices[start:start+batch_size]
            # dataset.data.shape = (1, 1000, 1, 64, 64), so batch_data = dataset.data[:, batch_idx, ...]
            yield dataset.data[:, batch_idx, ...]
    else:
        indices = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(dataset), batch_size):
            batch_idx = indices[start:start+batch_size]
            batch_data = dataset.data[batch_idx]
            batch_labels = dataset.labels[batch_idx]
            yield batch_data, batch_labels

def pfd_To_dataloader(
    pfds: any,
    targets: Optional[any] = None,
    feature: dict = None,
    batch_size: int = 64,
    shuffle: bool = False
) -> Union[pfd_Dataloader, List[pfd_Dataloader]]:
    if not feature:
        raise ValueError("Feature dictionary cannot be None or empty")
    
    key = list(feature.keys())[0]
    values = list(feature.values())[0]

    sim_harmonic = False

    data = pfd_to_data(pfds, targets, feature, denoise=True, sim_harmonic=sim_harmonic)

    if targets is not None:
        label = data[-1]
        data = data[:-1]
        
    else:
        label = None
    
    def create_dataloader(data, label=None):
        dataset = pfd_Dataset(data, label) if label is not None else pfd_Dataset(data)
        # dataset.data.shape = (1, 1000, 1, 64, 64), so batch_data = dataset.data[:, batch_idx, ...]
        return pfd_Dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )    
    
    if isinstance(values, list):
        #dl_list: List[pfd_Dataloader] = [create_dataloader(data[i], label) for i in range(len(values))]
        return [create_dataloader(d, label) for d in data]
    
    else:
        if targets is None:
            dl = create_dataloader(data, label)
        else:
            dl = create_dataloader(data[0], label)
        return dl