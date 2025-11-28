import os
import pandas as pd
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ToTensord,
    ScaleIntensityd
)
from typing import List, Dict, Union, Tuple

from config.us_config import (
    US_IMAGE_PATH, LABEL_CSV_PATH, 
    ROTATE_RANGE, FLIP_PROB
)
from config.us_radiomics_config import RADIOMICS_CSV_PATH


def create_fusion_data_dicts(
    image_dir: str, 
    radiomics_csv_path: str,
    label_csv_path: str,
    label_column_index: int = 1
) -> Tuple[List[Dict[str, Union[str, list, int]]], List[int]]:
    """
    Reads image paths, radiomics features, and labels, creating data dictionaries for MONAI.
    
    Args:
        image_dir: Directory for processed US images.
        radiomics_csv_path: Path to the normalized radiomics features CSV file.
        label_csv_path: Path to the labels CSV file.
        label_column_index: Column index for the label in the CSV.

    Returns:
        A tuple containing:
        - data_dicts: List of dicts with "image", "radiomics" list, and "label".
        - raw_labels: List of raw integer labels for StratifiedKFold.
    """
    
    # 1. Load Radiomics Features
    radiomics_df = pd.read_csv(radiomics_csv_path)
    radiomics_features = radiomics_df.iloc[:, 1:].values.tolist()
    
    # 2. Load Labels and Image Paths
    label_df = pd.read_csv(label_csv_path)
    raw_labels = label_df.iloc[:, label_column_index].tolist()
    image_list = sorted(os.listdir(image_dir))
    
    # 3. Create Data Dictionaries
    data_dicts = []
    min_len = min(len(image_list), len(radiomics_features), len(raw_labels))
    
    for i in range(min_len):
        image_path_full = os.path.join(image_dir, image_list[i])
        
        if os.path.exists(image_path_full):
            data_dicts.append({
                "image": image_path_full,
                "radiomics": radiomics_features[i],
                "label": int(raw_labels[i])  
            })
    
    final_raw_labels = [d["label"] for d in data_dicts]
    
    print(f"Successfully loaded {len(data_dicts)} US-Radiomics fusion data entries.")
    return data_dicts, final_raw_labels


def get_fusion_transforms() -> Compose:
    """
    Defines the MONAI transformations for US-Radiomics training/validation.
    Includes image loading/scaling and feature/label tensor conversion.
    
    Note: For fusion models, we often use the training set augmentation for both 
    train/val transforms if the trainer is designed to handle it, 
    but for simplicity and adherence to pinjie_us.py which didn't show augmentation, 
    we use a basic pipeline here. 
    (Augmentation should ideally be applied in the trainer during training epochs only, 
    using transforms from us_data_loader.py)
    """
    
    transforms = Compose([
        LoadImaged(keys=["image"]), 
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image"]),
        
        lambda x: {**x, "radiomics": torch.tensor(x["radiomics"], dtype=torch.float)},
        lambda x: {**x, "label": torch.tensor(x["label"] - 1, dtype=torch.long)},
    ])
    return transforms

if __name__ == '__main__':
    print("--- Testing US-Radiomics Fusion Data Loader ---")
    
    try:
        from config.us_config import US_IMAGE_PATH, LABEL_CSV_PATH
        data_dicts, raw_labels = create_fusion_data_dicts(
            US_IMAGE_PATH, RADIOMICS_CSV_PATH, LABEL_CSV_PATH
        )
        fusion_transforms = get_fusion_transforms()
        
        print(f"Created {len(data_dicts)} fusion data dictionaries.")
        if data_dicts:
            print(f"Sample dictionary keys: {list(data_dicts[0].keys())}")
            print(f"Sample radiomics feature length: {len(data_dicts[0]['radiomics'])}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure data paths are correct for testing.")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")