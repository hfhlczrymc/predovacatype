import os
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
import numpy as np
import time
from typing import List, Dict

from config.us_radiomics_config import (
    INITIAL_LR, WEIGHT_DECAY, MAX_EPOCHS, 
    EARLY_STOPPING_PATIENCE, BEST_METRIC_NAME, 
    US_RADIOMICS_MODEL_SAVE_PATH
)
from config.us_config import BATCH_SIZE

class USRadiomicsTrainer:
    """
    Core class for training the US-Radiomics Feature Fusion Model.
    Implements multimodal input handling, adaptive LR scheduling, and AUC-based early stopping.
    """
    def __init__(self, model: nn.Module):
        """Initializes the Trainer with model, device, loss, and optimizer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=INITIAL_LR,
            weight_decay=WEIGHT_DECAY
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5,
            verbose=True,
            min_lr=1e-8
        )
        self.best_metric = -float('inf')
        self.epochs_no_improve = 0

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Runs a single training epoch with multimodal input."""
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            # Inputs: image and radiomics features
            inputs_img = batch["image"].to(self.device)
            inputs_rad = batch["radiomics"].to(self.device)
            targets = batch["label"].to(self.device)
            
            outputs = self.model(inputs_img, inputs_rad)
            loss = self.criterion(outputs, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def _calculate_metrics(self, all_targets: np.ndarray, all_preds: np.ndarray, all_outputs_prob: np.ndarray) -> dict:
        """Calculates performance metrics (AUC, Acc, F1, Specificity, Sensitivity)."""
        
        val_acc = accuracy_score(all_targets, all_preds)
        try:
            val_auc = roc_auc_score(all_targets, all_outputs_prob)
        except ValueError:
            val_auc = 0.0
            
        val_f1 = f1_score(all_targets, all_preds)

        cm = confusion_matrix(all_targets, all_preds)
        
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            'accuracy': val_acc,
            'auc': val_auc,
            'f1_score': val_f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
        }

    def _validate_epoch(self, dataloader: DataLoader) -> dict:
        """Runs a single validation epoch and calculates metrics."""
        self.model.eval()
        all_targets = []
        all_outputs_prob = []
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                inputs_img = batch["image"].to(self.device)
                inputs_rad = batch["radiomics"].to(self.device)
                targets = batch["label"].to(self.device)
                
                outputs = self.model(inputs_img, inputs_rad)
                
                preds = torch.argmax(outputs, dim=1)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                
                all_targets.extend(targets.cpu().numpy())
                all_outputs_prob.extend(probabilities.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        metrics = self._calculate_metrics(
            np.array(all_targets), 
            np.array(all_preds), 
            np.array(all_outputs_prob)
        )
        
        return metrics

    def _save_checkpoint(self, fold_idx: int, epoch: int, metrics: dict, is_best: bool):
        """Saves the model state dictionary as a checkpoint."""
        
        filename = f"Fold_{fold_idx + 1}"
        
        if is_best:
            old_best_files = [f for f in os.listdir(US_RADIOMICS_MODEL_SAVE_PATH) if f.startswith(f"{filename}_best_model")]
            for old_file in old_best_files:
                os.remove(os.path.join(US_RADIOMICS_MODEL_SAVE_PATH, old_file))

            save_path = os.path.join(US_RADIOMICS_MODEL_SAVE_PATH, 
                                     f"{filename}_best_model_{BEST_METRIC_NAME}{metrics[BEST_METRIC_NAME]:.4f}"
                                     f"_acc{metrics['accuracy']:.4f}_f1{metrics['f1_score']:.4f}.pth")
                                     
            print(f"--- Saved BEST Model for Fold {fold_idx + 1} at Epoch {epoch + 1} ({BEST_METRIC_NAME}: {metrics[BEST_METRIC_NAME]:.4f}) ---")
        else:
            old_epoch_files = [f for f in os.listdir(US_RADIOMICS_MODEL_SAVE_PATH) if f.startswith(f"{filename}_epoch")]
            for old_file in old_epoch_files:
                os.remove(os.path.join(US_RADIOMICS_MODEL_SAVE_PATH, old_file))
                
            save_path = os.path.join(US_RADIOMICS_MODEL_SAVE_PATH, f"{filename}_epoch_{epoch + 1}.pth")

        torch.save(self.model.state_dict(), save_path)


    def run_fold(
        self, 
        fold_idx: int, 
        train_data: List[dict], 
        val_data: List[dict], 
        transforms: nn.Module
    ):
        """Runs the complete training and validation cycle for one fold."""
        
        self.best_metric = -float('inf')
        self.epochs_no_improve = 0
        
        train_dataset = Dataset(data=train_data, transform=transforms)
        val_dataset = Dataset(data=val_data, transform=transforms) 
        
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"\n{'='*50}\nStarting US-Radiomics Fusion Training for Fold {fold_idx + 1}...\n{'='*50}")

        for epoch in range(MAX_EPOCHS):
            start_time = time.time()
            
            # 1. Training step
            train_loss = self._train_epoch(train_dataloader)
            
            # 2. Validation step
            val_metrics = self._validate_epoch(val_dataloader)
            current_metric = val_metrics[BEST_METRIC_NAME] 
            
            # 3. Learning Rate Scheduling
            self.scheduler.step(current_metric)
            
            # 4. Logging
            elapsed_time = time.time() - start_time
            print(f"Fold {fold_idx + 1} | Epoch {epoch + 1}/{MAX_EPOCHS} ({elapsed_time:.2f}s) | "
                  f"Loss: {train_loss:.6f} | Acc: {val_metrics['accuracy']:.4f} | AUC: {val_metrics['auc']:.4f} | "
                  f"Sen: {val_metrics['sensitivity']:.4f} | Spe: {val_metrics['specificity']:.4f} | F1: {val_metrics['f1_score']:.4f}")

            # 5. Early Stopping and Checkpoint Saving
            is_best = False
            if epoch >= 5 and current_metric > self.best_metric: 
                self.best_metric = current_metric
                self.epochs_no_improve = 0
                is_best = True
                
            elif epoch >= 5:
                self.epochs_no_improve += 1
            
            self._save_checkpoint(fold_idx, epoch, val_metrics, is_best)
            
            if self.epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered for Fold {fold_idx + 1} after {EARLY_STOPPING_PATIENCE} epochs of no improvement.")
                break
        
        print(f"Fold {fold_idx + 1} Training Finished. Best {BEST_METRIC_NAME}: {self.best_metric:.4f}")