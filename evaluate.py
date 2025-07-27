"""
Usage:
python evaluate.py | tee evaluate_result.txt
"""
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
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_utils import read_pfd_label
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tinygrad.helpers import colored
import glob

val_dataloader = load_from_pickle('./dataset/val_dataloader.pkl')
test_dataloader = load_from_pickle('./dataset/test_dataloader.pkl')

combineModel = CombinedModel()
features = {
    'profiles': {'phasebins': [64, 96, 128]},
    'intervals': {'intervals': [64, 96, 128]}
}
multi_AI = Multi_AI(combineModel, features, ckpt="./trained_model/weight_0.9954_0.9830.pth")

print(colored(f"\n== Val dataset results ==",'blue'))

val_preds_prob = multi_AI.classifier.predict_prob(val_dataloader)


telescope_data = {
    "Parkes": [],
    "FAST": [],
    "Arecibo": []
}

X_pfd, y_label = read_pfd_label('./dataset/val_labels.txt')

for line in zip(X_pfd, y_label, val_preds_prob):
    path, label, prob = line
    label = int(label)
    prob = float(prob)
    for telescope in telescope_data:
        if telescope in path:
            telescope_data[telescope].append((label, prob))
            break

for telescope, data in telescope_data.items():
    if not data:
        print(f"{telescope}: No data found.\n")
        continue

    y_true = np.array([x[0] for x in data])
    y_prob = np.array([x[1] for x in data])
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred) * 100
    eval_recall = recall_score(y_true, y_pred, average=None)
    eval_precision = precision_score(y_true, y_pred, average=None)
    eval_f1 = f1_score(y_true, y_pred, average=None)
    _, recall_pos = eval_recall
    _, precision_pos = eval_precision
    _, f1_pos = eval_f1

    print(f"== {telescope} ==")
    print(f"Accuracy : {acc:.2f}%")
    print(f"Precision: {precision_pos*100:.2f}%")
    print(f"Recall   : {recall_pos*100:.2f}%")
    print(f"F1 Score : {f1_pos*100:.2f}%\n")

print(colored(f"\n== MWA Test dataset results ==",'blue'))

test_preds = multi_AI.classifier.predict(test_dataloader)
X_pfd, y_label = read_pfd_label('./dataset/test_labels.txt')
y_label = np.array(y_label, dtype=int)

eval_recall = recall_score(y_label, test_preds , average=None)
eval_precision = precision_score(y_label, test_preds , average=None)
eval_f1 = f1_score(y_label, test_preds , average=None)
# eval_f1 = f1_score(y_label, test_preds , average='macro')
recall_neg, recall_pos = eval_recall
precision_neg, precision_pos = eval_precision
f1_neg, f1_pos = eval_f1
cm = confusion_matrix(y_label, test_preds )
tn, fp, fn, tp = cm.ravel()

print(colored(f"Positive class: recall={recall_pos:.4f}, precision={precision_pos:.4f}, F1-score={f1_pos:.4f}", "yellow"))
print(colored(f"Negative class: recall={recall_neg:.4f}, precision={precision_neg:.4f}, F1-score={f1_neg:.4f}", "yellow"))

print(f"TN (True Negative: negative samples correctly identified as negative):    {tn}")
print(f"FP (False Positive: negative samples incorrectly identified as positive): {fp}")
print(f"FN (False Negative: positive samples incorrectly identified as negative): {fn}")
print(f"TP (True Positive: positive samples correctly identified as positive):    {tp}")

print(colored(f"\n== Finetuned MWA Test dataset results ==",'blue'))

multi_AI = Multi_AI(combineModel, features, ckpt="./trained_model/finetune_mwa_weight.pth")

test_preds = multi_AI.classifier.predict(test_dataloader)
X_pfd, y_label = read_pfd_label('./dataset/test_labels.txt')
y_label = np.array(y_label, dtype=int)

eval_recall = recall_score(y_label, test_preds , average=None)
eval_precision = precision_score(y_label, test_preds , average=None)
eval_f1 = f1_score(y_label, test_preds , average=None)
# eval_f1 = f1_score(y_label, test_preds , average='macro')
recall_neg, recall_pos = eval_recall
precision_neg, precision_pos = eval_precision
f1_neg, f1_pos = eval_f1
cm = confusion_matrix(y_label, test_preds )
tn, fp, fn, tp = cm.ravel()

print(colored(f"Positive class: recall={recall_pos:.4f}, precision={precision_pos:.4f}, F1-score={f1_pos:.4f}", "yellow"))
print(colored(f"Negative class: recall={recall_neg:.4f}, precision={precision_neg:.4f}, F1-score={f1_neg:.4f}", "yellow"))

print(f"TN (True Negative: negative samples correctly identified as negative):    {tn}")
print(f"FP (False Positive: negative samples incorrectly identified as positive): {fp}")
print(f"FN (False Negative: positive samples incorrectly identified as negative): {fn}")
print(f"TP (True Positive: positive samples correctly identified as positive):    {tp}")

print(colored(f"\n== GBNCC dataset results ==",'blue'))

multi_AI = Multi_AI(combineModel, features, ckpt="./trained_model/weight_0.9954_0.9830.pth")

pfdlist = glob.glob('../data/Multi_AI/GBNCC/pulsars/*.pfd')
GBNCC_pulsars_pred = multi_AI.predict_prob(pfdlist)
GBNCC_pulsars_5 = len(np.where(GBNCC_pulsars_pred > 0.5)[0])
GBNCC_pulsars_3 = len(np.where(GBNCC_pulsars_pred > 0.3)[0])

pfdlist = glob.glob('../data/Multi_AI/GBNCC/harmonic/*.pfd')
GBNCC_harmonic_pred = multi_AI.predict_prob(pfdlist)
GBNCC_harmonic_5 = len(np.where(GBNCC_harmonic_pred > 0.5)[0])
GBNCC_harmonic_3 = len(np.where(GBNCC_harmonic_pred > 0.3)[0])

pfdlist = glob.glob('../data/Multi_AI/GBNCC/nonpulsars/*.pfd')
GBNCC_nonpulsars_pred = multi_AI.predict_prob(pfdlist)
GBNCC_nonpulsars_5 = len(np.where(GBNCC_nonpulsars_pred > 0.5)[0])
GBNCC_nonpulsars_3 = len(np.where(GBNCC_nonpulsars_pred > 0.3)[0])

print(f"Pulsars - Threshold 0.5: {GBNCC_pulsars_5}, 0.3: {GBNCC_pulsars_3}")
print(f"Harmonic - Threshold 0.5: {GBNCC_harmonic_5}, 0.3: {GBNCC_harmonic_3}")
print(f"Nonpulsars - Threshold 0.5: {GBNCC_nonpulsars_5}, 0.3: {GBNCC_nonpulsars_3}")

print(colored(f"\n== NGC 5904 (M5) results ==",'blue'))

pfdlist = glob.glob('../data/Multi_AI/M5_candidates/*.pfd')
M5_pred = multi_AI.predict_prob(pfdlist)
M5_results = len(np.where(M5_pred > 0.5)[0])
print(f"NGC 5904 (M5) total candidates: {len(M5_pred)}, predict prob > 0.5: {M5_results}")

print(colored(f"\n== FAST simulate pulsar results ==",'blue'))

pfdlist = glob.glob('../data/Multi_AI/FAST_sim/FAST_sim_20/*.pfd')
FAST_pred = multi_AI.predict_prob(pfdlist)
FAST_results = len(np.where(FAST_pred > 0.5)[0])
print(f"FAST simulate pulsar total candidates: {len(FAST_pred)}, predict prob > 0.5: {FAST_results}")