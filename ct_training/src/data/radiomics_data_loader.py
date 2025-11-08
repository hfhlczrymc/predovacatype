import os
import pandas as pd
import torch
import numpy as np

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ToTensord,
)
from typing import List, Dict, Union, Tuple

# Import configurations
from config.ct_config import CT_IMAGE_PATH, LABEL_CSV_PATH
from config.radiomics_config import RADIOMICS_CSV_PATH

def create_fusion_data_dicts(
    ct_image_dir: str, 
    radiomics_csv_path: str,
    label_csv_path: str,
    label_column_index: int = 1
) -> Tuple[List[Dict[str, Union[str, np.ndarray, int]]], List[int]]:
    """
    Reads image paths, Radiomics features, and labels, creating data dictionaries.

    Args:
        ct_image_dir: Directory for processed CT images.
        radiomics_csv_path: Path to the normalized Radiomics features CSV.
        label_csv_path: Path to the labels CSV file.
        label_column_index: Column index for the label in the CSV.

    Returns:
        A tuple containing:
        - data_dicts: List of dicts with "image" path, "radiomics" features, and "label".
        - raw_labels: List of raw integer labels for StratifiedKFold.
    """
    if not all(os.path.exists(p) for p in [ct_image_dir, radiomics_csv_path, label_csv_path]):
        raise FileNotFoundError("One or more required data files/directories not found.")

    # 1. Load Labels and Radiomics Features
    label_df = pd.read_csv(label_csv_path)
    radiomics_df = pd.read_csv(radiomics_csv_path)

    # Assuming patient IDs (filenames) align in sorted order across the image folder, 
    # the label CSV, and the radiomics CSV (a critical assumption from the source code).
    ct_list = sorted(os.listdir(ct_image_dir))
    
    # Extract raw labels (1/2)
    raw_labels = label_df.iloc[:, label_column_index].tolist()
    
    # Extract radiomics feature vectors (skipping the first column, which is usually patient ID)
    radiomics_features = []
    for i in range(len(radiomics_df)):
        # Assuming the CSV contains the Patient ID in the first column (index 0)
        features = radiomics_df.iloc[i, 1:].values.astype(np.float32)
        radiomics_features.append(features)

    # 2. Match and Create Data Dictionaries
    data_dicts = []
    
    # The lowest count of items determines the size of the final dataset
    min_len = min(len(ct_list), len(radiomics_features), len(raw_labels))
    
    for i in range(min_len):
        ct_path_full = os.path.join(ct_image_dir, ct_list[i])
        
        # Check if image file exists before adding to data_dicts
        if os.path.exists(ct_path_full):
            data_dicts.append({
                "image": ct_path_full,
                "radiomics": radiomics_features[i],
                "label": int(raw_labels[i])  
            })
    
    # Update the labels list to match the final data_dicts list size and order
    final_raw_labels = [d["label"] for d in data_dicts]
    
    print(f"Successfully loaded {len(data_dicts)} fusion data entries.")
    return data_dicts, final_raw_labels

def get_fusion_transforms() -> Compose:
    """Defines the MONAI transformations for the fusion model (image and radiomics)."""
    
    transforms = Compose([
        # Image loading and channel handling
        LoadImaged(keys=["image"]), 
        EnsureChannelFirstd(keys=["image"]),
        
        # Custom transformation to adjust the label from 1/2 to 0/1
        lambda x: {**x, "label": torch.tensor(x["label"] - 1, dtype=torch.long)}, 
        
        # Radiomics feature conversion
        # The radiomics feature list (numpy array) must be converted to torch.float
        lambda x: {**x, "radiomics": torch.tensor(x["radiomics"], dtype=torch.float)},
        
        # Final image tensor conversion
        ToTensord(keys=["image"]),
    ])
    return transforms

if __name__ == '__main__':
    # Example usage and testing (requires actual data files)
    print("--- Testing Fusion Data Loader ---")
    
    try:
        # Note: These paths should point to actual data for real execution
        fusion_dicts, raw_labels = create_fusion_data_dicts(
            CT_IMAGE_PATH, RADIOMICS_CSV_PATH, LABEL_CSV_PATH
        )
        fusion_transforms = get_fusion_transforms()
        
        print(f"Created {len(fusion_dicts)} fusion data dictionaries.")
        if fusion_dicts:
            print(f"Sample dictionary keys: {list(fusion_dicts[0].keys())}")
            print(f"Sample radiomics shape: {fusion_dicts[0]['radiomics'].shape}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure data paths in config files are correct for testing.")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")