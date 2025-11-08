import torch
import torch.nn as nn
from monai.networks.nets import resnet34
from typing import Tuple

# Import architecture dimensions
from config.radiomics_config import (
    RADIOMICS_FEATURE_DIM, DL_FEATURE_DIM, FC1_OUTPUT_DIM, 
    FC2_INPUT_DIM, FC2_OUTPUT_DIM, NUM_CLASSES
)

class FeatureFusionModel(nn.Module):
    """
    A multimodal model that fuses Deep Learning features from a pre-trained ResNet-34 
    with Radiomics features via a set of fully connected layers.
    """
    def __init__(self, pretrained_model_path: str):
        super().__init__()
        
        # 1. Image Feature Extractor (ResNet-34)
        base_model = resnet34(
            pretrained=False, 
            spatial_dims=3, 
            n_input_channels=1, 
            num_classes=NUM_CLASSES
        )
        
        # Load the pre-trained weights from the image-only model
        if not pretrained_model_path or not torch.exists(pretrained_model_path):
             raise FileNotFoundError(f"Pretrained model not found at: {pretrained_model_path}")

        base_model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'))
        
        # Extract features before the final classification layer
        self.feature_extractor = nn.Sequential(*(list(base_model.children())[:-1]))
        
        # Freeze the weights of the feature extractor (as implied by the original code's intent)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # 2. Radiomics Feature MLP
        self.radiomics_mlp = nn.Sequential(
            nn.Linear(RADIOMICS_FEATURE_DIM, FC1_OUTPUT_DIM),
            nn.ReLU()
        )
        
        # 3. Fusion Feature MLP (DL_feature + Radiomics_feature_512)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(FC2_INPUT_DIM, FC2_OUTPUT_DIM),
            nn.ReLU(),
            nn.Linear(FC2_OUTPUT_DIM, NUM_CLASSES)
        )

    def forward(self, input_img: torch.Tensor, input_radiomics: torch.Tensor) -> torch.Tensor:

        dl_feature = self.feature_extractor(input_img).flatten(1)
        radiomics_feature = self.radiomics_mlp(input_radiomics)
        concat_feature = torch.cat((dl_feature, radiomics_feature), dim=1)
        outputs = self.fusion_mlp(concat_feature)
        
        return outputs