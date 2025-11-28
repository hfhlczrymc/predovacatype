import torch
from torch import nn
import torchvision.models as models

from config.us_config import N_SPLITS 

def get_us_model() -> nn.Module:
    """
    Creates and modifies the 2D ResNet-34 model for single-channel US image classification.
    """
    model = models.resnet34(weights=None)    
    model.conv1 = nn.Conv2d(
        in_channels=1, 
        out_channels=64, 
        kernel_size=7, 
        stride=2, 
        padding=3, 
        bias=False
    )    
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5), 
        nn.Linear(256, 2) 
    )
    
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
    model.conv1.apply(init_weights)
    model.fc.apply(init_weights)
    
    return model

if __name__ == '__main__':
    model = get_us_model()
    print("US Model Architecture (2D ResNet-34 modified):")
    print(model)
    dummy_input = torch.randn(8, 1, 256, 256) 
    dummy_output = model(dummy_input)
    print(f"\nTest input shape: {dummy_input.shape}")
    print(f"Test output shape: {dummy_output.shape}") 