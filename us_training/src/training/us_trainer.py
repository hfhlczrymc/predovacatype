import os
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import time
from typing import List, Dict

# Import configurations
from config.us_config import (
    INITIAL_LR, WEIGHT_DECAY, MAX_EPOCHS, 
    EARLY_STOPPING_PATIENCE, BEST_METRIC_NAME, 
    METRIC_CHECK_START_EPOCH, US_MODEL_SAVE_PATH, 
    RANDOM_SEED
)

class USTrainer:
    """
    Core class for training the 2D Ultrasound (US) model.
    Implements learning rate scheduling and AUC-based early stopping.
    """
    def __init__(self, model: nn.Module):
        """Initializes the Trainer with model, device, loss, and optimizer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = Adam(
            self.model.parameters(), 
            lr=INITIAL_LR,
            weight_decay=WEIGHT_DECAY
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        self.best_metric = -float('inf')
        self.epochs_no_improve = 0
        torch.manual_seed(RANDOM_SEED)

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            inputs = batch["image"].to(self.device)
            targets = batch["label"].to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def _validate_epoch(self, dataloader: DataLoader) -> dict:
        """Runs a single validation epoch and calculates metrics."""
        self.model.eval()
        all_targets = []
        all_outputs_prob = []
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["image"].to(self.device)
                targets = batch["label"].to(self.device)
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                probabilities = torch.softmax(outputs, dim=1)[:, 1] 
                all_targets.extend(targets.cpu().numpy())
                all_outputs_prob.extend(probabilities.cpu().numpy())
                total_correct += (preds == targets).sum().item()
                total_samples += targets.size(0)

        val_acc = total_correct / total_samples
        try:
            val_auc = roc_auc_score(all_targets, all_outputs_prob)
        except ValueError:
            val_auc = 0.0

        return {
            'accuracy': val_acc,
            'auc': val_auc,
        }

    def _save_checkpoint(self, fold_idx: int, epoch: int, metrics: dict, is_best: bool):
        """Saves the model state dictionary as a checkpoint."""
        
        filename = f"Fold_{fold_idx + 1}"
        current_metric_value = metrics[BEST_METRIC_NAME]
        
        if is_best:
            old_best_files = [f for f in os.listdir(US_MODEL_SAVE_PATH) if f.startswith(f"{filename}_best_model")]
            for old_file in old_best_files:
                os.remove(os.path.join(US_MODEL_SAVE_PATH, old_file))

            save_path = os.path.join(US_MODEL_SAVE_PATH, 
                                     f"{filename}_best_model_{BEST_METRIC_NAME}{current_metric_value:.4f}_acc{metrics['accuracy']:.4f}.pth")
            print(f"--- Saved BEST Model for Fold {fold_idx + 1} at Epoch {epoch + 1} ({BEST_METRIC_NAME}: {current_metric_value:.4f}) ---")
        else:
            old_epoch_files = [f for f in os.listdir(US_MODEL_SAVE_PATH) if f.startswith(f"{filename}_epoch")]
            for old_file in old_epoch_files:
                os.remove(os.path.join(US_MODEL_SAVE_PATH, old_file))
                
            save_path = os.path.join(US_MODEL_SAVE_PATH, f"{filename}_epoch_{epoch + 1}.pth")

        torch.save(self.model.state_dict(), save_path)


    def run_fold(
        self, 
        fold_idx: int, 
        train_data: List[dict], 
        val_data: List[dict], 
        train_transforms: nn.Module, 
        val_transforms: nn.Module, 
        batch_size: int
    ) -> float:
        """Runs the complete training and validation cycle for one fold."""
        
        self.best_metric = -float('inf')
        self.epochs_no_improve = 0
        self.model.apply(lambda m: self.model.apply(m.init_weights) if hasattr(m, 'init_weights') else None)

        train_dataset = Dataset(data=train_data, transform=train_transforms)
        val_dataset = Dataset(data=val_data, transform=val_transforms)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"\n{'='*50}\nStarting US Training for Fold {fold_idx + 1}...\n{'='*50}")

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
                  f"Train Loss: {train_loss:.6f} | Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f}")

            # 5. Early Stopping and Checkpoint Saving
            is_best = False
            if epoch >= METRIC_CHECK_START_EPOCH and current_metric > self.best_metric:
                self.best_metric = current_metric
                self.epochs_no_improve = 0
                is_best = True
                
            elif epoch >= METRIC_CHECK_START_EPOCH:
                self.epochs_no_improve += 1
            
            self._save_checkpoint(fold_idx, epoch, val_metrics, is_best)
            
            # Check for early stopping condition
            if self.epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered for Fold {fold_idx + 1} after {EARLY_STOPPING_PATIENCE} epochs of no improvement.")
                break
        
        return self.best_metric