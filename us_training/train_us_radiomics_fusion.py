import torch
import numpy as np
import random
import os
import glob
import time
from sklearn.model_selection import StratifiedKFold
from typing import List, Dict

from config.us_config import RANDOM_SEED 
from config.us_config import US_IMAGE_PATH, LABEL_CSV_PATH 
from config.us_radiomics_config import (
    RADIOMICS_CSV_PATH, N_SPLITS, 
    PRETRAINED_MODEL_DIR, PRETRAINED_MODEL_FILENAME_FORMAT, BATCH_SIZE
)

from src.data.us_radiomics_data_loader import create_fusion_data_dicts, get_fusion_transforms
from src.models.us_feature_fusion_model import USFeatureFusionModel
from src.training.us_radiomics_trainer import USRadiomicsTrainer

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_best_pretrained_model_path(fold_idx: int) -> str:
    """Dynamically finds the best pre-trained US model checkpoint for the current fold."""

    search_pattern = PRETRAINED_MODEL_FILENAME_FORMAT.format(fold_num=fold_idx + 1)
    full_pattern = os.path.join(PRETRAINED_MODEL_DIR, search_pattern)
    
    best_models = glob.glob(full_pattern)
    
    if not best_models:
        raise FileNotFoundError(f"Could not find best pre-trained US model for Fold {fold_idx + 1} at {full_pattern}. "
                                "Ensure train_us.py has been executed successfully.")
    
    return best_models[0]

def main():
    """Main function to perform five-fold cross-validation training for the US-Radiomics fusion model."""
    set_seed(RANDOM_SEED)
    print(f"Random seed set to {RANDOM_SEED}.")
    start_time_total = time.time()

    # 1. Load Data Dictionaries
    data_dicts, raw_labels = create_fusion_data_dicts(
        US_IMAGE_PATH, RADIOMICS_CSV_PATH, LABEL_CSV_PATH
    )

    if not data_dicts:
        print("Error: No US-Radiomics fusion data entries loaded. Exiting training.")
        return

    # 2. Define Transforms
    fusion_transforms = get_fusion_transforms()

    # 3. Setup Cross-Validation
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    folds = list(kf.split(range(len(data_dicts)), raw_labels))
    
    # 4. Run Training for Each Fold
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        
        print(f"\n--- Preparing US-Radiomics Fusion Model for Fold {fold_idx + 1} ---")

        try:
            pretrained_path = get_best_pretrained_model_path(fold_idx)
        except FileNotFoundError as e:
            print(e)
            continue
        model = USFeatureFusionModel(pretrained_model_path=pretrained_path)
        for name, param in model.model0.named_parameters():
            param.requires_grad = False
        print(f"Base US ResNet-34 weights for Fold {fold_idx + 1} are now frozen.")
        trainer = USRadiomicsTrainer(model)
        train_data = [data_dicts[i] for i in train_idx]
        val_data = [data_dicts[i] for i in val_idx]

        trainer.run_fold(
            fold_idx=fold_idx, 
            train_data=train_data, 
            val_data=val_data, 
            transforms=fusion_transforms
        )

    time_elapsed_total = time.time() - start_time_total
    print("\n" + "="*70)
    print(f"US-Radiomics Feature Fusion Cross-Validation Complete. Total time: {time_elapsed_total:.2f}s")
    print("="*70)

if __name__ == "__main__":
    main()