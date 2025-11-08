import torch
from monai.networks.nets import resnet34
from torch import nn
from typing import Final

from config.ct_config import SPATIAL_DIMS, N_INPUT_CHANNELS, NUM_CLASSES, MODEL_NAME

def get_ct_model() -> nn.Module:
    """
    Initializes and returns the 3D ResNet-34 model for CT image classification.
    
    The architecture uses the three-dimensional variant for CT, consistent with the methodology.

    Returns:
        A torch.nn.Module instance of the 3D ResNet-34 model.
    """
    print(f"Initializing model: {MODEL_NAME} (3D variant)")
    
    model = resnet34(
        pretrained=False, 
        spatial_dims=SPATIAL_DIMS, 
        n_input_channels=N_INPUT_CHANNELS, 
        num_classes=NUM_CLASSES 
    )    
    return model

if __name__ == '__main__':
    print("--- Testing Model Initialization ---")
    ct_model = get_ct_model()
    
    dummy_input_shape = (1, N_INPUT_CHANNELS, 8, 256, 256) 
    dummy_input = torch.randn(dummy_input_shape)
    
    try:
        ct_model.eval()
        with torch.no_grad():
            output = ct_model(dummy_input)
            
        expected_output_shape = (1, NUM_CLASSES)
        
        print(f"Model initialized successfully.")
        print(f"Input shape: {dummy_input_shape}")
        print(f"Output shape: {tuple(output.shape)}")
        
        if tuple(output.shape) == expected_output_shape:
            print("Output shape matches the expected class number.")
        else:
            print(f"ERROR: Output shape mismatch. Expected {expected_output_shape}, got {tuple(output.shape)}")

    except Exception as e:
        print(f"An error occurred during model testing: {e}")