from tinygrad import Tensor, nn
from typing import List, Optional, Tuple
import numpy as np
from multiprocessing import Pool, cpu_count
from data_loader import pfdreader
from dataset import pfd_To_dataloader
from models.cnn_attention import CNN_Attention
from models.se_resnet10_1D import SE_ResNet10_1D
from models.combined_model import CombinedModel
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import datetime, random
from tinygrad import Device
from tinygrad.nn import optim
from tinygrad.helpers import getenv, colored, trange
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
import yaml

def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config

global_config = load_config('./config/global_cfg.yaml')

Device.DEFAULT = global_config['DEVICE']

predict_batch_size = global_config['PREDICTION']['BATCH_SIZE']

use_multiprocessing = global_config['MULTIPROCESSING']['ENABLED']

if use_multiprocessing:
    num_workers = global_config['MULTIPROCESSING']['NUM_WORKERS']

import math

class ReduceLROnPlateau:
    # Reduce learning rate when a metric has stopped improving.
    def __init__(self, optimizer, factor=0.5, patience=5, min_lr=1e-6):
        """
        Args:
            optimizer: Optimizer with a 'lr' attribute to be updated.
            factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor.
            patience (int): Number of epochs with no improvement after which learning rate will be reduced.
            min_lr (float): Lower bound on the learning rate.
        """
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = float('-inf')
        self.num_bad_epochs = 0

    def step(self, metric):
        if metric > self.best:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
            if new_lr < self.optimizer.lr:
                self.optimizer.lr = new_lr
                self.num_bad_epochs = 0

class CosineAnnealingScheduler:
    """
    Cosine annealing learning rate scheduler.
    Decreases the learning rate following a cosine curve from the initial value to a minimum.
    """
    def __init__(self, optimizer, max_epochs, min_lr=1e-6):
        """
        Args:
            optimizer: Optimizer with a 'lr' attribute to be updated.
            max_epochs (int): Total number of epochs for annealing.
            min_lr (float): Minimum learning rate at the end of the schedule.
        """
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.initial_lr = optimizer.lr

    def step(self, current_epoch):
        lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * current_epoch / self.max_epochs))
        self.optimizer.lr = lr

def focal_loss(logits, target, alpha=0.75, gamma=2.0):
    """
    Compute the focal loss for multi-class classification.

    Args:
        logits (Tensor): Raw model outputs of shape (batch_size, num_classes).
        target (Tensor): Ground truth labels of shape (batch_size,), with integer class indices.
        alpha (float): Balancing factor for class imbalance.
        gamma (float): Focusing parameter to down-weight easy examples.
    """
    probs = logits.softmax()
    target_one_hot = Tensor.eye(logits.shape[-1])[target]
    pt = (probs * target_one_hot).sum(axis=1)
    loss = -alpha * (1 - pt) ** gamma * pt.log()
    return loss.mean()

class Multi_AI:
    def __init__(self, combineModel: CombinedModel, features: dict = None, ckpt: Optional[str] = None):
        """
        Args:
            combineModel: Multi-input classification
            features:   {
                        'profiles': {'phasebins': [64, 96, 128]},     # profile feature
                        'intervals': {'intervals': [64, 96, 128]}     # time vs phase feature
                        }
            ckpt: Path to a checkpoint file. If provided, loads model weights from it.
        """
        self.model = combineModel
        self.classifier = Classifier(self.model)
        self.features = features
        if ckpt is not None:
            self.classifier.load_ckpt(ckpt)

    def create_Combined_dataloader(self, pfds: List[str], labels=None, batch_size=256) -> List[Tuple]:
        # Create a combined dataloader for multi-input models using .pfd files.
        if isinstance(pfds[0], pfdreader):
            pfdfile = pfds
        else:
            with Pool(processes=num_workers) as pool:
                pfdfile = list(tqdm(pool.imap(pfdreader, pfds),
                total=len(pfds),
                desc='Loading PFDs',
                mininterval=1.0))

        def get_dataloaders(feature: str):
            return pfd_To_dataloader(
                pfdfile,
                labels,
                feature=feature,
                batch_size=batch_size,
                shuffle=False
            )
        # name_1: 'profiles'; feature_1: {'phasebins': [64, 96, 128]}
        # name_2: 'intervals'; feature_2: 'intervals': {'intervals': [64, 96, 128]}
        dataloaders = dict({name: get_dataloaders(feature) for name, feature in self.features.items()})

        combined_dataloaders = list(zip(
            dataloaders['profiles'][0], dataloaders['profiles'][1], dataloaders['profiles'][2],
            dataloaders['intervals'][0], dataloaders['intervals'][1], dataloaders['intervals'][2]
        ))

        """
        def combined_gen():
            for a, b, c, d, e, f in zip(
                dataloaders['profiles'][0],
                dataloaders['profiles'][1],
                dataloaders['profiles'][2],
                dataloaders['intervals'][0],
                dataloaders['intervals'][1],
                dataloaders['intervals'][2]
            ):
                yield [a, b, c, d, e, f]  
        combined_dataloaders = combined_gen()
        """
        return combined_dataloaders

    def set(self, loss_func,
            optimizer=None,
            scheduler=None,
            max_epochs: int = 10,
            ckpt_path: str = './checkpoints', # ckpt_path: Path to save model checkpoints.
            ) -> None:  
        self.classifier.configure(
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            max_epochs=max_epochs,
            ckpt_path=ckpt_path
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        train_dataloader = self.create_Combined_dataloader(X_train, y_train)
        val_dataloader = None
        test_dataloader = None

        if X_val is not None and y_val is not None:
            val_dataloader = self.create_Combined_dataloader(X_val, y_val)
        if X_test is not None and y_test is not None:
            test_dataloader = self.create_Combined_dataloader(X_test, y_test)

        self.classifier.fit(train_dataloader, val_dataloader, test_dataloader)

    def fit_dataloader(self, train_dataloader, val_dataloader, test_dataloader):
        self.classifier.fit(train_dataloader, val_dataloader, test_dataloader)

    def predict(self, pfds: List[str], outfile: Optional[str] = None):
        dataloader = self.create_Combined_dataloader(pfds, batch_size=predict_batch_size)
        results = self.classifier.predict(dataloader)
        if outfile:
            self._save_results(pfds, results, outfile)
        return results

    def predict_prob(self, pfds: List[str], outfile: Optional[str] = None):
        dataloader = self.create_Combined_dataloader(pfds, batch_size=predict_batch_size)
        results = self.classifier.predict_prob(dataloader)
        if outfile:
            self._save_results(pfds, results, outfile)
        return results

    def test_datasets(self, X_test, y_test):
        test_dataloader = self.create_Combined_dataloader(X_test, y_test)
        self.classifier.test_datasets(test_dataloader)

    def _save_results(self, pfds: List[str], results: List[str], outfile: str) -> None:
        with open(outfile, 'w') as f:
            for pfd, result in zip(pfds, results):
                f.write(f"{pfd}\t{result}\n")

    def finetune(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None, ckpt_path='./finetune_checkpoints'):

        train_dataloader = self.create_Combined_dataloader(X_train, y_train)
        val_dataloader = None
        test_dataloader = None

        if X_val is not None and y_val is not None:
            val_dataloader = self.create_Combined_dataloader(X_val, y_val)
        if X_test is not None and y_test is not None:
            test_dataloader = self.create_Combined_dataloader(X_test, y_test)

        params_fine_tune = []
        for name, param in get_state_dict(self).items():
            if any(x in name for x in ["model.classifier", "classifier.net.classifier", "optimizer", "scheduler"]):
                params_fine_tune.append(param)
            else:
                param.requires_grad = False
                params_fine_tune.append(param)
        
        fine_tune_opt = optim.AdamW(params_fine_tune, lr=self.classifier.optimizer.lr.numpy()[0], eps=self.classifier.optimizer.eps, weight_decay=self.classifier.optimizer.wd)

        fine_tune_scheduler = CosineAnnealingScheduler(fine_tune_opt, max_epochs=self.classifier.max_epochs)

        self.set(
            loss_func=self.classifier.loss_func,
            optimizer=fine_tune_opt,
            scheduler=fine_tune_scheduler,
            max_epochs=self.classifier.max_epochs,
            ckpt_path=ckpt_path
        )

        self.classifier.fit(train_dataloader, val_dataloader, test_dataloader)

    def finetune_dataloader(self, train_dataloader, val_dataloader=None, test_dataloader=None, ckpt_path='./finetune_checkpoints'):

        params_fine_tune = []
        for name, param in get_state_dict(self).items():
            if any(x in name for x in ["model.classifier", "classifier.net.classifier", "optimizer", "scheduler"]):
                params_fine_tune.append(param)
            else:
                param.requires_grad = False
                params_fine_tune.append(param)
        
        fine_tune_opt = optim.AdamW(params_fine_tune, lr=self.classifier.optimizer.lr.numpy()[0], eps=self.classifier.optimizer.eps, weight_decay=self.classifier.optimizer.wd)

        fine_tune_scheduler = CosineAnnealingScheduler(fine_tune_opt, max_epochs=self.classifier.max_epochs)

        self.set(
            loss_func=self.classifier.loss_func,
            optimizer=fine_tune_opt,
            scheduler=fine_tune_scheduler,
            max_epochs=self.classifier.max_epochs,
            ckpt_path=ckpt_path
        )

        self.classifier.fit(train_dataloader, val_dataloader, test_dataloader)


class Classifier:
    def __init__(self, net):
        self.net = net

    def configure(self, loss_func, optimizer, scheduler, max_epochs: int = 20, ckpt_path: str = './checkpoints'):
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.ckpt_path = ckpt_path
        self.scheduler = scheduler

        self.train_loss = np.zeros(max_epochs)
        self.train_acc = np.zeros(max_epochs)
        self.train_precision = np.zeros(max_epochs)
        self.train_recall = np.zeros(max_epochs)
        self.train_f1 = np.zeros(max_epochs)

        self.val_loss = np.zeros(max_epochs)
        self.val_acc = np.zeros(max_epochs)
        self.val_precision = np.zeros(max_epochs)
        self.val_recall = np.zeros(max_epochs)
        self.val_f1 = np.zeros(max_epochs)

    def __call__(self, *inputs) -> Tensor:
        return self.net(*inputs)

    def fit(self, dl_train, dl_val=None, dl_test=None):
        print("[INFO] Start training")
        self.print_bar()

        for epoch in range(self.max_epochs):
            
            if self.scheduler != None:
                self.scheduler.step(epoch)

            total_batches = len(dl_train)
            t = trange(total_batches, desc=f"Epoch {epoch+1}/{self.max_epochs}")
            
            self.train_epoch(dl_train, epoch, t, shuffle=True)

            if dl_val:
                self.validate_epoch(dl_val, epoch)

            if (self.val_acc[epoch] >= max(self.val_acc)):
                self.save_ckpt(f"{self.ckpt_path}/epoch{epoch+1}_{self.train_acc[epoch]:.4f}_{self.val_acc[epoch]:.4f}.pth")

            print(f"Epoch {epoch + 1}/{self.max_epochs}\n",
                  colored(f"train: loss={self.train_loss[epoch]:.4f} acc={self.train_acc[epoch]:.4f} recall={self.train_recall[epoch]:.4f} pre={self.train_precision[epoch]:.4f} f1={self.train_f1[epoch]:.4f}\n", "cyan"),
                  colored(f" val: loss={self.val_loss[epoch]:.4f} acc={self.val_acc[epoch]:.4f} recall={self.val_recall[epoch]:.4f} pre={self.val_precision[epoch]:.4f} f1={self.val_f1[epoch]:.4f}", "cyan"))
            
            if dl_test:
                self.test_datasets(dl_test)

        self.print_bar()
        print("[INFO] End training")

    @Tensor.train()
    def train_epoch(self, dl_train, epoch, t, shuffle=True):
        total_loss, correct, total = 0.0, 0, 0
        all_preds = None
        all_labels = None

        if shuffle:
            random.shuffle(dl_train)

        for batch in dl_train:
            loss, corr, count, preds, labels = self.train_batch(batch)
            t.update(1)
            t.set_description(f"Loss: {loss.item():6.4f}")
            total_loss += loss.item()
            correct += corr.item()
            total += count
            if all_preds is None:
                all_preds = preds  
                all_labels = labels
            else:
                all_preds = all_preds.cat(preds, dim=0)
                all_labels = all_labels.cat(labels, dim=0)

        all_preds = all_preds.numpy()
        all_labels = all_labels.numpy()
        train_recall = recall_score(all_labels, all_preds, average=None)
        train_precision = precision_score(all_labels, all_preds, average=None)
        train_f1 = f1_score(all_labels, all_preds, average=None)
        _, recall_pos = train_recall
        _, precision_pos = train_precision
        _, f1_pos = train_f1
            
        self.train_loss[epoch] = total_loss / len(dl_train)
        self.train_acc[epoch] = correct / total
        self.train_recall[epoch] = recall_pos
        self.train_precision[epoch] =  precision_pos
        self.train_f1[epoch] = f1_pos
        

    @Tensor.test()
    def validate_epoch(self, dl_val, epoch):
        total_loss, correct, total = 0.0, 0, 0
        all_preds = None
        all_labels = None

        for batch in dl_val:
            loss, corr, count, labels, preds = self.validate_batch(batch)
            total_loss += loss.item()
            correct += corr.item()
            total += count

            if all_preds is None:
                all_preds = preds  
                all_labels = labels
            else:
                all_preds = all_preds.cat(preds, dim=0)
                all_labels = all_labels.cat(labels, dim=0)

        all_preds = all_preds.numpy()
        all_labels = all_labels.numpy()
        eval_recall = recall_score(all_labels, all_preds, average=None)
        eval_precision = precision_score(all_labels, all_preds, average=None)
        eval_f1 = f1_score(all_labels, all_preds, average=None)
        _, recall_pos = eval_recall
        _, precision_pos = eval_precision
        _, f1_pos = eval_f1

        self.val_loss[epoch] = total_loss / len(dl_val)
        self.val_acc[epoch] = correct / total
        self.val_recall[epoch] = recall_pos
        self.val_precision[epoch] =  precision_pos
        self.val_f1[epoch] = f1_pos

    @Tensor.train()
    def train_batch(self, batch):
        inputs = [b[0] for b in batch] # 6 features x batches x (64,96,128)
        inputs = list(map(Tensor, inputs))
        labels = Tensor([b[1] for b in batch][0]) # batches x labels

        self.optimizer.zero_grad()
        out = self(*inputs)
        loss = self.loss_func(out, labels).backward()

        self.optimizer.step()

        preds = out.argmax(axis=1)
        correct = (preds == labels).sum()
        total = labels.shape[0]

        return loss, correct, total, labels, preds
    
    @Tensor.test()
    def validate_batch(self, batch):
        inputs = [b[0] for b in batch] # 6 features x batches x (64,96,128)
        inputs = list(map(Tensor, inputs))
        labels = Tensor([b[1] for b in batch][0]) # batches x labels

        out = self(*inputs)
        loss = self.loss_func(out, labels)

        preds = out.argmax(axis=1)
        correct = (preds == labels).sum()
        total = labels.shape[0]

        return loss, correct, total, labels, preds

    @Tensor.test()
    def predict(self, dl):
        results = None
        total_batches = len(dl)
        t = trange(total_batches)
        for batch in dl:
            inputs = [b[0] for b in batch] # 6 features x batches x (64,96,128)
            inputs = list(map(Tensor, inputs))
            out = self(*inputs)
            preds = out.argmax(axis=1)
            t.update(1)
            if results is None:
                results = preds  
            else:
                results = results.cat(preds, dim=0)
        
        return results.numpy()

    @Tensor.test()
    def predict_prob(self, dl):
        results = None
        total_batches = len(dl)
        t = trange(total_batches)
        for batch in dl:
            inputs = [b[0] for b in batch] # 6 features x batches x (64,96,128)
            inputs = list(map(Tensor, inputs))
            out = self(*inputs)
            probs = out.softmax(axis=1)[:, 1]
            t.update(1)
            if results is None:
                results = probs 
            else:
                results = results.cat(probs, dim=0)
        return results.numpy()


    @Tensor.test()
    def test_datasets(self, test_dataloader):
        total_loss, correct, total = 0.0, 0, 0
        all_preds = None
        all_labels = None

        for batch in test_dataloader:
            loss, corr, count, labels, preds = self.validate_batch(batch)
            total_loss += loss.item()
            correct += corr.item()
            total += count
            if all_preds is None:
                all_preds = preds  
                all_labels = labels
            else:
                all_preds = all_preds.cat(preds, dim=0)
                all_labels = all_labels.cat(labels, dim=0)

        eval_loss = total_loss / len(test_dataloader)
        eval_acc = correct / total
        all_preds = all_preds.numpy()
        all_labels = all_labels.numpy()
        eval_recall = recall_score(all_labels, all_preds, average=None)
        eval_precision = precision_score(all_labels, all_preds, average=None)
        eval_f1 = f1_score(all_labels, all_preds, average=None)
        # eval_f1 = f1_score(all_labels, all_preds, average='macro')
        recall_neg, recall_pos = eval_recall
        precision_neg, precision_pos = eval_precision
        f1_neg, f1_pos = eval_f1
        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()

        print(colored(f"Test results: loss={eval_loss:.4f}, accuracy={eval_acc:.4f}", "yellow"))
        print(colored(f"Positive class: recall={recall_pos:.4f}, precision={precision_pos:.4f}, F1-score={f1_pos:.4f}", "yellow"))
        print(colored(f"Negative class: recall={recall_neg:.4f}, precision={precision_neg:.4f}, F1-score={f1_neg:.4f}", "yellow"))

        print(f"TN (True Negative: negative samples correctly identified as negative):    {tn}")
        print(f"FP (False Positive: negative samples incorrectly identified as positive): {fp}")
        print(f"FN (False Negative: positive samples incorrectly identified as negative): {fn}")
        print(f"TP (True Positive: positive samples correctly identified as positive):    {tp}")

    def save_ckpt(self, path: str = "model_weights.pth"):
        params = get_state_dict(self.net)
        safe_save(params, path)
        print(colored(f"Model weights saved to {path}", "green"))

    def load_ckpt(self, path: str = "model_weights.pth"):
        state_dict = safe_load(path)
        load_state_dict(self.net, state_dict)
        print(colored(f"Model weights loaded from {path}", "green"))

    @staticmethod
    def print_bar():
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "=" * 80 + f" {nowtime}")

    def print_train(self):
        print("\nTrain dataset stat print::")
        for i in range(self.max_epochs):
            print(f"Epoch {i+1} loss={self.train_loss[i]:.4f} acc={self.train_acc[i]:.4f} recall={self.train_recall[i]:.4f}")

    def print_val(self):
        print("\nVal dataset stat print:")
        for i in range(self.max_epochs):
            print(f"Epoch {i+1} loss={self.val_loss[i]:.4f} acc={self.val_acc[i]:.4f} recall={self.val_recall[i]:.4f}")