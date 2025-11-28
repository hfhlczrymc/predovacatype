import os
import pandas as pd
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandRotated,
    RandFlipd,
    ToTensord,
)
from typing import List, Dict, Union, Tuple

# Import configurations
from config.us_config import (
    US_IMAGE_PATH, LABEL_CSV_PATH, 
    ROTATE_RANGE, FLIP_PROB
)


def create_data_dicts(
    image_dir: str, 
    label_csv_path: str,
    label_column_index: int = 1
) -> Tuple[List[Dict[str, Union[str, int]]], List[int]]:
    """
    Reads image paths and labels, creating data dictionaries for MONAI.
    
    Args:
        image_dir: Directory for processed US images.
        label_csv_path: Path to the labels CSV file.
        label_column_index: Column index for the label in the CSV.

    Returns:
        A tuple containing:
        - data_dicts: List of dicts with "image" path and "label".
        - raw_labels: List of raw integer labels for StratifiedKFold.
    """
    if not all(os.path.exists(p) for p in [image_dir, label_csv_path]):
        raise FileNotFoundError("One or more required data files/directories not found.")

    label_df = pd.read_csv(label_csv_path)
    image_list = sorted(os.listdir(image_dir))
    
    raw_labels = label_df.iloc[:, label_column_index].tolist()
    
    data_dicts = []
    min_len = min(len(image_list), len(raw_labels))
    
    for i in range(min_len):
        image_path_full = os.path.join(image_dir, image_list[i])
        if os.path.exists(image_path_full):
            data_dicts.append({
                "image": image_path_full,
                "label": int(raw_labels[i])  
            })
    final_raw_labels = [d["label"] for d in data_dicts]
    
    print(f"Successfully loaded {len(data_dicts)} US data entries.")
    return data_dicts, final_raw_labels


def get_train_transforms() -> Compose:
    """Defines the MONAI transformations for US training (with augmentation)."""
    
    transforms = Compose([
        LoadImaged(keys=["image"]), 
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]), 
        RandRotated(
            keys=["image"],
            range_x=ROTATE_RANGE, 
            prob=0.5,
            padding_mode="border"
        ),
        RandFlipd(keys=["image"], spatial_axis=1, prob=FLIP_PROB),  
        lambda x: {**x, "label": torch.tensor(x["label"], dtype=torch.long)}, 
        ToTensord(keys=["image"]),
    ])
    return transforms


def get_validation_transforms() -> Compose:
    """Defines the MONAI transformations for US validation (without augmentation)."""
    
    transforms = Compose([
        LoadImaged(keys=["image"]), 
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]), 
        lambda x: {**x, "label": torch.tensor(x["label"], dtype=torch.long)}, 
        ToTensord(keys=["image"]),
    ])
    return transforms

if __name__ == '__main__':
    # Example usage and testing (requires actual data files)
    print("--- Testing US Data Loader ---")
    
    try:
        data_dicts, raw_labels = create_data_dicts(US_IMAGE_PATH, LABEL_CSV_PATH)
        train_transforms = get_train_transforms()
        val_transforms = get_validation_transforms()
        print(f"Created {len(data_dicts)} data dictionaries.")
        if data_dicts:
            print(f"Sample dictionary keys: {list(data_dicts[0].keys())}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure data paths in config files are correct for testing.")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")