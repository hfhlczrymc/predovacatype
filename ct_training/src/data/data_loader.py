import os
import pandas as pd
import torch
import numpy as np

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ToTensord,
    RandRotate90d,
    RandFlipd,
    Spacingd,
)
from typing import List, Dict, Union, Tuple

from config.ct_config import (
    CT_IMAGE_PATH, 
    LABEL_CSV_PATH, 
    MAX_ROTATION_DEGREES, 
    DO_HORIZONTAL_FLIP, 
    RANDOM_SEED
)

def create_data_dicts(
    ct_image_dir: str, 
    label_csv_path: str,
    label_column_index: int = 1
) -> Tuple[List[Dict[str, Union[str, int]]], List[int]]:
    """
    Reads image files and labels to create MONAI-compatible data dictionaries.

    Args:
        ct_image_dir: Directory containing the processed CT image files.
        label_csv_path: Path to the CSV file containing the labels.
        label_column_index: Column index (0-based) for the label in the CSV.

    Returns:
        A tuple containing:
        - data_dicts: List of dictionaries with "image" path and "label".
        - labels: List of raw integer labels for StratifiedKFold.
    """
    if not os.path.exists(ct_image_dir):
        raise FileNotFoundError(f"CT image directory not found: {ct_image_dir}")
    if not os.path.exists(label_csv_path):
        raise FileNotFoundError(f"Label CSV file not found: {label_csv_path}")

    label_df = pd.read_csv(label_csv_path)
    ct_list = sorted(os.listdir(ct_image_dir))
    labels = label_df.iloc[:, label_column_index].tolist()
    
    if len(ct_list) != len(labels):
         print(f"Warning: Number of images ({len(ct_list)}) does not match number of labels ({len(labels)}).")
    
    data_dicts = []
    for ct_file, label in zip(ct_list, labels):
        ct_path_full = os.path.join(ct_image_dir, ct_file)
        if os.path.exists(ct_path_full):
            data_dicts.append({
                "image": ct_path_full,
                "label": int(label)  
            })
    
    final_labels = [d["label"] for d in data_dicts]
    
    print(f"Successfully loaded {len(data_dicts)} data entries.")
    return data_dicts, final_labels

def get_train_transforms() -> Compose:
    """Defines the MONAI transformations for the training dataset, including data augmentation."""
    
    from monai.transforms import RandRotated, RandFlipd
    
    transforms = Compose([
        LoadImaged(keys=["image"]), 
        EnsureChannelFirstd(keys=["image"]),
        RandRotated(
            keys=["image"],
            range_x=MAX_ROTATION_DEGREES * (np.pi/180),  
            range_y=MAX_ROTATION_DEGREES * (np.pi/180),
            range_z=MAX_ROTATION_DEGREES * (np.pi/180),
            prob=0.5,
            padding_mode="border",
            spatial_reorient=True,
            
        ),
        RandFlipd(keys=["image"], spatial_axis=0, prob=0.5) if DO_HORIZONTAL_FLIP else lambda x: x,
        lambda x: {**x, "label": torch.tensor(x["label"], dtype=torch.long)}, 
        ToTensord(keys=["image"]),
    ])
    return transforms

def get_validation_transforms() -> Compose:
    """Defines the MONAI transformations for the validation dataset (no augmentation)."""
    
    transforms = Compose([
        LoadImaged(keys=["image"]), 
        EnsureChannelFirstd(keys=["image"]),
        lambda x: {**x, "label": torch.tensor(x["label"], dtype=torch.long)},
        ToTensord(keys=["image"]),
    ])
    return transforms

if __name__ == '__main__':
    print("--- Testing Data Loader ---")
    try:
        data_dicts, raw_labels = create_data_dicts(CT_IMAGE_PATH, LABEL_CSV_PATH)
        train_transforms = get_train_transforms()
        val_transforms = get_validation_transforms()
        
        print(f"Created {len(data_dicts)} data dictionaries.")
        print(f"Sample dictionary: {data_dicts[0]}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure data paths in ct_config.py are correct for testing.")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")