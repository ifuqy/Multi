from tinygrad import Tensor, nn
from tinygrad.nn import optim
from tinygrad.nn.state import safe_save
from tinygrad.helpers import trange
import numpy as np
import matplotlib.pyplot as plt
from models.Basic_UNet import BasicUNet  
from data_loader import dataloader
import random
import os
from tinygrad import Device
from utils import load_from_pickle
import yaml
from sklearn.utils import shuffle
# Device.DEFAULT = "NV:1"

# As Tensor.test() decorator was removed in v0.11.0, return empty decorator if test method doesn't exist
def safe_decorator(cls, attr_name, *args, **kwargs):
    deco = getattr(cls, attr_name, None)
    if deco is not None:
        return deco(*args, **kwargs)
    return lambda f: f

# === Utility Functions ===

def psnr(x: Tensor, y: Tensor) -> float:
    max_val = max(x.max().item(), y.max().item())  # get the maximum value dynamically
    min_val = min(x.min().item(), y.min().item())
    peak = max_val - min_val  # dynamic range
    mse = ((x - y)**2).mean().item()
    return 20 * np.log10(peak / np.sqrt(mse + 1e-8))

def ssim_loss(x: Tensor, y: Tensor) -> Tensor:
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = ((x - mu_x) ** 2).mean()
    sigma_y = ((y - mu_y) ** 2).mean()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    max_val = x.max().maximum(y.max())
    min_val = x.min().minimum(y.min())
    L = max_val - min_val  # dynamic range

    C1 = (0.01*L) ** 2
    C2 = (0.03*L) ** 2

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

    return 1 - ssim_map  # SSIM loss = 1 - SSIM


def gradient_loss(pred: Tensor, target: Tensor) -> Tensor:
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    # dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]

    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]

    # dx_loss = (dx_pred - dx_target).abs().mean()
    dy_loss = (dy_pred - dy_target).abs().mean()

    return 0.5 * dy_loss + 0.5 * (1.0 - dx_pred.abs().mean())
    #return 0.5 * dx_loss + 0.5 * dy_loss

    
def loss_fn(pred: Tensor, target: Tensor) -> Tensor:
    mse = ((pred - target) ** 2).mean()
    ssim = ssim_loss(pred, target)
    edge = gradient_loss(pred, target)
    return 0.6 * mse + 0.2 * ssim + 0.2 * edge

def get_noise_level(epoch, max_epoch, batch_size, min_t=500, max_t=999, beta=2.0):
    """
    Use the Beta distribution to make the noise more gentle in the early stages and increase rapidly in the later stages.
    beta > 1 means "more small noise in the early stage", becoming more aggressive only in the later stages.
    beta=2~5: The larger the value, the more "conservative" in the early stage, making it easier to converge.
    beta=1 is equivalent to linear growth.
    beta < 1 results in aggressiveness in the early stage and is not recommended.
    """
    ratio = epoch / max_epoch
    scale = ratio ** beta  # nonlinear increase
    current_max = int(min_t + (max_t - min_t) * scale)
    return np.random.randint(min_t, current_max + 1, size=(batch_size,))


def visualize_denoising(x_vis, noisy_vis, pred_vis, epoch: int, save_dir="denoise_vis"):
    """
    Save denoising visualization results for 4 random samples from the batch.
    """
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(4, x_vis.shape[0])):
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        axs[0].imshow(x_vis[i].numpy()[0], cmap='gray')
        axs[0].set_title("Clean")
        axs[1].imshow(noisy_vis[i].numpy()[0], cmap='gray')
        axs[1].set_title("Noisy")
        axs[2].imshow(pred_vis[i].numpy()[0], cmap='gray')
        axs[2].set_title("Denoised")
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/epoch_{epoch}_sample_{i}.png", dpi=150)
        plt.close()


def plot_loss_curve(losses, total_batches, filename="./denoise_trained/loss_curve.png"):
    """
    Plot training loss curve using Nature-style aesthetics.
    """
    losses = np.array(losses)
    epochs = np.arange(len(losses)) / total_batches

    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.linewidth': 1.0,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
    })

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(epochs, losses, color='black', linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Training Loss (MSE)", fontsize=12)
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.close()

def plot_psnr_curve(psnr_val, filename="./denoise_trained/psnr_curve.png"):
    """
    Plot training loss curve using Nature-style aesthetics.
    """
    psnr_val = np.array(psnr_val)
    epochs = np.arange(len(psnr_val))
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 5,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
    })

    fig, ax = plt.subplots(figsize=(2.5, 1.8))
    ax.plot(epochs, psnr_val, color='black', linewidth=0.6)
    ax.set_xlabel("Epoch", fontsize=5)
    ax.set_ylabel("PSNR (dB)", fontsize=5)
    y_min = 23
    y_max = int(psnr_val.max()) + 1 
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(y_min, y_max + 1, 1))

    ax.set_xlim(-1, len(psnr_val) + 0.5)
    ax.set_xticks(np.arange(0, len(psnr_val) + 0.5, 5))  
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.close()

# === Dataset Loader ===

class denoise_Dataset:
    def __init__(self, pfds):
        self.pfds = pfds

    def __len__(self):
        return len(self.pfds)

    def __getitem__(self, idx):
        return Tensor(self.pfds[idx].getdata({'intervals': 64}))  # shape (1, 64, 64)
    
def split_dataset(dataset, val_size=400):
    all_indices = np.arange(len(dataset))
    val_indices = np.random.choice(all_indices, size=val_size, replace=False)
    train_indices = np.setdiff1d(all_indices, val_indices)
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    return train_dataset, val_dataset

# === Add Noise Schedule ===

def add_noise(x: Tensor, noise: Tensor, t: np.ndarray) -> Tensor:
    """
    Apply noise to input x using linear alpha schedule.
    """
    alpha_t = 1.0 - t / 1000.0
    alpha_t = alpha_t[:, None, None, None]
    alpha_t = Tensor(alpha_t.astype(np.float32))
    return x * alpha_t.sqrt() + noise * (1 - alpha_t).sqrt()

if __name__ == "__main__":

    # === Load Configuration ===
    with open('./config/denoise_model_cfg.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Device Configuration
    device_default = Device.DEFAULT
    if config['device'] != "Default":
        Device.DEFAULT = config['device']

    # === Load and Prepare Dataset ===

    # Data Loading Configuration
    data_config = config['data']
    if data_config['source'] == 'pkl':
        # Load from pickle file
        train_pfds = load_from_pickle(data_config['pfds_path'])
        train_target = load_from_pickle(data_config['target_path'])
    else:  # text file
        # Load from text file (custom implementation required)
        path = data_config['txt_path']
        loader = dataloader(path)
        train_pfds = loader.pfds 
        train_target = loader.target
        # shuffle witch random_state
        train_pfds, train_target = shuffle(
            train_pfds, train_target, random_state=int(data_config['random_state'])
        )

    # Dataset Processing
    ldf = train_pfds
    labels = train_target
    indices = np.where(labels == 1)[0]
    pfds = np.array(ldf)[indices]
    dataset = denoise_Dataset(pfds)

    # Evaluate PSNR on 400 random samples
    train_dataset, val_dataset = split_dataset(
        dataset, 
        val_size=config['hyperparameters']['val_size']
    )

    # === Hyperparameters ===
    hp = config['hyperparameters']
    batch_size = hp['batch_size']
    n_epochs = hp['n_epochs']
    lr = hp['lr']
    weight_decay = hp['weight_decay']

    # Training Tracking
    losses = []
    psnr_avg_vals = []
    psnr_vals_all = []

    # Print Configuration Parameters
    print("\n=== Configuration Parameters ===")
    print(f"Data Source: {data_config['source']}")
    print(f"Training Data Shape: {pfds.shape}")
    print(f"Shuffle Status: Data has been shuffled")

    print("\nHyperparameters:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Learning Rate: {lr}")
    print(f"  Weight Decay: {weight_decay}")
    print(f"  Validation Size: {hp['val_size']}")

    print(f"\nDefault Device: {device_default}")
    print(f"Use Device: {Device.DEFAULT}")
    print("=========================\n")

    # === Model Setup ===

    net = BasicUNet()
    opt = optim.AdamW(nn.state.get_parameters(net), lr=lr, weight_decay=weight_decay)

    @Tensor.train()
    def train_batch(x_batch: Tensor, noisy_x: Tensor) -> Tensor:
        opt.zero_grad()
        pred = net(noisy_x)
        # loss = ((pred - x_batch)**2).mean()
        loss = loss_fn(pred, x_batch)
        #loss, mse, ssim, edge = loss_fn(pred, x_batch)
        loss.backward()
        opt.step()
        # print(f"SSIM: {ssim.item():.4f} | Gradient: {edge.item():.4f}")
        return loss

    @safe_decorator(Tensor, "test")
    def net_output(x: Tensor) -> Tensor:
        return net(x)

    # === Training Loop ===

    best_loss = float('inf')
    best_psnr = 0.0

    for epoch in range(1, n_epochs + 1):
        perm = np.random.permutation(len(train_dataset))
        total_batches = len(train_dataset) // batch_size

        for b in trange(total_batches, desc=f"Epoch {epoch}/{n_epochs}"):
            batch_idx = perm[b * batch_size : (b + 1) * batch_size]
            batch_samples = [train_dataset[i] for i in batch_idx]
            x_batch = Tensor.stack(*batch_samples)

            noise = Tensor.randn(*x_batch.shape)
            t = get_noise_level(epoch, n_epochs, x_batch.shape[0])
            noisy_x = add_noise(x_batch, noise, t)

            loss = train_batch(x_batch, noisy_x)
            losses.append(loss.item())

        avg_loss = sum(losses[-total_batches:]) / total_batches
        print(f"Epoch {epoch} finished. Avg loss: {avg_loss:.6f}")

        random.shuffle(val_dataset)
        x_vis = Tensor.stack(*val_dataset)

        sample_batch_size = 100
        psnr_vals = []

        for i in range(0, len(x_vis), sample_batch_size):
            x_sub = x_vis[i:i+sample_batch_size]
            noise_sub = Tensor.randn(*x_sub.shape)
            t_sub = get_noise_level(epoch, n_epochs, x_sub.shape[0])
            noisy_sub = add_noise(x_sub, noise_sub, t_sub)
            pred_sub = net_output(noisy_sub)
            psnr_val = psnr(pred_sub, x_sub)
            psnr_vals.append(psnr_val)
            psnr_vals_all.append(psnr_val)
        psnr_avg = sum(psnr_vals)/len(psnr_vals)
        psnr_avg_vals.append(psnr_avg)

        print(f"Epoch {epoch} PSNR: {psnr_avg:.2f} dB")
        
        # Save 4 sample visualizations
        visualize_denoising(x_sub[:4], noisy_sub[:4], pred_sub[:4], epoch)

        # Save best model based on avg loss and PSNR
        if avg_loss < best_loss or psnr_avg > best_psnr:
            best_loss = min(avg_loss, best_loss)
            best_psnr = max(psnr_avg, best_psnr)
            safe_save(nn.state.get_state_dict(net), f"./denoise_trained/denoise_model_epoch{epoch}_psnr{best_psnr:.2f}dB.pth")
            print(f"✓ Model saved at epoch {epoch} (Best loss: {best_loss:.4f}, PSNR: {best_psnr:.2f})")

    # === Plot and Save Final Loss Curve ===
    plot_loss_curve(losses, total_batches)
    plot_psnr_curve(psnr_avg_vals)
