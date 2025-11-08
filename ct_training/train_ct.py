import torch
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
import time

from config.ct_config import (
    RANDOM_SEED, CT_IMAGE_PATH, LABEL_CSV_PATH, 
    N_SPLITS, BATCH_SIZE
)

from src.data.data_loader import (
    create_data_dicts, 
    get_train_transforms, 
    get_validation_transforms
)
from src.models.ct_model import get_ct_model
from src.training.trainer import CTTrainer

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    """Main function to perform five-fold cross-validation training."""
    set_seed(RANDOM_SEED)
    print(f"Random seed set to {RANDOM_SEED}.")
    start_time_total = time.time()

    # 1. Load Data Dictionaries and Labels
    data_dicts, raw_labels = create_data_dicts(CT_IMAGE_PATH, LABEL_CSV_PATH)

    if not data_dicts:
        print("Error: No data entries loaded. Exiting training.")
        return

    # 2. Define Transforms
    train_transforms = get_train_transforms()
    val_transforms = get_validation_transforms()

    # 3. Setup Cross-Validation
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    folds = list(kf.split(range(len(data_dicts)), raw_labels))
    
    # 4. Run Training for Each Fold
    for fold_idx, (train_idx, val_idx) in enumerate(folds):       
        model = get_ct_model()
        trainer = CTTrainer(model)
        train_data = [data_dicts[i] for i in train_idx]
        val_data = [data_dicts[i] for i in val_idx]
        trainer.run_fold(
            fold_idx=fold_idx, 
            train_data=train_data, 
            val_data=val_data, 
            train_transforms=train_transforms, 
            val_transforms=val_transforms, 
            batch_size=BATCH_SIZE
        )
    time_elapsed_total = time.time() - start_time_total
    print("\n" + "="*50)
    print(f"Cross-Validation Training Complete. Total time: {time_elapsed_total:.2f}s")
    print("="*50)

if __name__ == "__main__":
    main()