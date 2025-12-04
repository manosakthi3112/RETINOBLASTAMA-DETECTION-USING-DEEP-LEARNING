# -*- coding: utf-8 -*-
"""
EfficientNetV2-M binary classification with:
1. Training + validation split
2. Augmentations
3. Loss & accuracy tracking
4. Saving best model (atomic save)
5. Saving model at every epoch with val loss in filename (atomic save)
6. Plotting curves for loss, accuracy, precision, recall, F1, AUC
7. Saving summary stats and confusion matrices
"""

# ===================================================================
# 1. IMPORTS AND SETUP
# ===================================================================
import os
import random
import copy
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import datasets, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

# ===================================================================
# 2. CONFIGURATION
# ===================================================================
class CFG:
    DATA_DIR = './Final'
    MODEL_OUTPUT_DIR = './models'
    seed = 999
    img_size = (224, 224)
    epochs = 10
    batch_size = 12
    learning_rate = 1e-4
    train_val_split = 0.8
    num_workers = 0  # safer on Windows; increase if on Linux

# ===================================================================
# 3. UTILITY FUNCTIONS
# ===================================================================
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

def safe_torch_save(state_dict, path, use_new_zipfile_serialization=False):
    tmp_path = path + ".tmp"
    try:
        torch.save(state_dict, tmp_path, _use_new_zipfile_serialization=use_new_zipfile_serialization)
        os.replace(tmp_path, path)
    except Exception as e:
        try:
            torch.save(state_dict, tmp_path, _use_new_zipfile_serialization=not use_new_zipfile_serialization)
            os.replace(tmp_path, path)
        except Exception as e2:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            raise RuntimeError(f"Failed to save model to {path}: {e} | fallback: {e2}")

# ===================================================================
# 4. MODEL DEFINITION
# ===================================================================
class EffnetModel(nn.Module):
    def __init__(self):
        super().__init__()
        effnet = torchvision.models.efficientnet_v2_m(
            weights=torchvision.models.EfficientNet_V2_M_Weights.DEFAULT
        )
        self.feature_extractor = create_feature_extractor(effnet, ['flatten'])
        self.classifier = nn.Linear(1280, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)['flatten']
        output = self.classifier(features)
        return output

# ===================================================================
# 5. TRAINING AND VALIDATION LOGIC
# ===================================================================
def train_model(model, criterion, optimizer, dataloaders, device, num_epochs=10):
    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_precision": [], "val_recall": [],
        "val_f1": [], "val_auc": []
    }

    os.makedirs(CFG.MODEL_OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(CFG.MODEL_OUTPUT_DIR, "training_summary.txt")

    with open(summary_path, "w") as summary_file:
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                running_samples = 0
                all_labels, all_preds, all_probs = [], [], []

                loader = dataloaders[phase]
                for inputs, labels in tqdm(loader, desc=f"{phase.capitalize()} Phase", leave=False):
                    inputs = inputs.to(device)
                    labels = labels.to(device).float().unsqueeze(1)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        probs = torch.sigmoid(outputs)
                        preds = (probs > 0.5).float()

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    batch_size = labels.size(0)
                    running_loss += loss.item() * batch_size
                    running_corrects += torch.sum(preds == labels).item()
                    running_samples += batch_size

                    # ✅ FIXED: detach before converting to NumPy
                    all_labels.extend(labels.detach().cpu().numpy().flatten())
                    all_preds.extend(preds.detach().cpu().numpy().flatten())
                    all_probs.extend(probs.detach().cpu().numpy().flatten())

                epoch_loss = running_loss / running_samples
                epoch_acc = running_corrects / running_samples
                history[f"{phase}_loss"].append(epoch_loss)
                history[f"{phase}_acc"].append(epoch_acc)

                print(f'{phase.capitalize()} Loss: {epoch_loss:.6f} Acc: {epoch_acc:.4f}')

                if phase == 'val':
                    try:
                        precision = precision_score(all_labels, all_preds, zero_division=0)
                        recall = recall_score(all_labels, all_preds, zero_division=0)
                        f1 = f1_score(all_labels, all_preds, zero_division=0)
                        auc = roc_auc_score(all_labels, all_probs)
                    except Exception as e:
                        precision, recall, f1, auc = (0, 0, 0, 0)
                        print(f"Metric computation error: {e}")

                    history["val_precision"].append(precision)
                    history["val_recall"].append(recall)
                    history["val_f1"].append(f1)
                    history["val_auc"].append(auc)

                    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
                          f"F1: {f1:.4f}, AUC: {auc:.4f}")

                    summary_file.write(
                        f"Epoch {epoch+1} {phase} - Loss: {epoch_loss:.6f}, Acc: {epoch_acc:.4f}, "
                        f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}\n"
                    )
                    summary_file.flush()

                    cm = confusion_matrix(all_labels, all_preds)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    disp.plot(cmap='Blues', values_format='d')
                    plt.title(f'Confusion Matrix - Epoch {epoch+1}')
                    plt.savefig(os.path.join(
                        CFG.MODEL_OUTPUT_DIR,
                        f'confusion_matrix_epoch_{epoch+1}.png'
                    ))
                    plt.close()

                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        best_path = os.path.join(
                            CFG.MODEL_OUTPUT_DIR,
                            f'best_model_epoch_{epoch+1}_acc_{best_acc:.4f}.pth'
                        )
                        safe_torch_save(model.state_dict(), best_path)
                        print(f"New best model saved to {best_path}")

            val_loss_for_filename = history["val_loss"][-1]
            epoch_model_path = os.path.join(
                CFG.MODEL_OUTPUT_DIR,
                f'epoch_{epoch+1}_valLoss_{val_loss_for_filename:.4f}.pth'
            )
            safe_torch_save(model.state_dict(), epoch_model_path)
            print(f"Model for epoch {epoch+1} saved to {epoch_model_path}")

        time_elapsed = time.time() - start_time
        summary_file.write(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n")
        summary_file.write(f"Best val Acc: {best_acc:.4f}\n")

    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)

    # ===================================================================
    # 6. PLOTS
    # ===================================================================
    def save_plot(values_dict, title, ylabel, filename):
        plt.figure()
        for key, values in values_dict.items():
            plt.plot(values, label=key)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.title(title)
        plt.savefig(os.path.join(CFG.MODEL_OUTPUT_DIR, filename))
        plt.close()

    save_plot(
        {"Train Loss": history["train_loss"], "Val Loss": history["val_loss"]},
        "Loss Curve", "Loss", "loss_curve.png"
    )
    save_plot(
        {"Train Acc": history["train_acc"], "Val Acc": history["val_acc"]},
        "Accuracy Curve", "Accuracy", "accuracy_curve.png"
    )
    save_plot(
        {
            "Precision": history["val_precision"],
            "Recall": history["val_recall"],
            "F1": history["val_f1"],
            "AUC": history["val_auc"]
        },
        "Validation Metrics", "Score", "validation_metrics_curve.png"
    )

    return model

# ===================================================================
# 7. MAIN EXECUTION SCRIPT
# ===================================================================
def main():
    seed_everything(CFG.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(CFG.img_size),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(CFG.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    print("Loading dataset...")
    full_dataset = datasets.ImageFolder(CFG.DATA_DIR)
    if len(full_dataset) == 0:
        raise RuntimeError(f"No images found in {CFG.DATA_DIR}. Check your DATA_DIR.")
    indices = list(range(len(full_dataset)))
    train_size = int(CFG.train_val_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, subset: Subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            x, y = self.subset[idx]
            if self.transform:
                x = self.transform(x)
            return x, y

    train_dataset = TransformedDataset(train_dataset, data_transforms['train'])
    val_dataset = TransformedDataset(val_dataset, data_transforms['val'])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers),
        'val': DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
    }

    print(f"Classes: {full_dataset.classes}")
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    model = EffnetModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG.learning_rate)

    print("\nStarting training...")
    trained_model = train_model(model, criterion, optimizer, dataloaders, device, num_epochs=CFG.epochs)
    print("\nTraining finished!")

if __name__ == '__main__':
    main()
