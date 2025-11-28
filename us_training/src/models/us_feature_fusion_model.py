import torch
from torch import nn
import torchvision.models as models
from typing import Optional

# Import configurations
from config.us_radiomics_config import RADIOMICS_FEATURE_DIM

class USFeatureFusionModel(nn.Module):
    """
    Model for fusing 2D US image features (from ResNet-34) and Radiomics features.
    It loads a pre-trained US model, extracts deep features via a hook, 
    concatenates them with processed radiomics features, and uses a fusion head for classification.
    """
    def __init__(self, pretrained_model_path: Optional[str] = None):
        """
        Initializes the fusion model.

        Args:
            pretrained_model_path: Path to the best checkpoint of the base US model.
        """
        super().__init__()
        
        # 1. Initialize the Base 2D ResNet-34 Model Structure
        model0 = models.resnet34(weights=None)
        model0.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model0.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        
        # 2. Load Pre-trained Weights
        if pretrained_model_path:
            print(f"Loading pre-trained US weights from: {pretrained_model_path}")
            model0.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'))
        
        self.model0 = model0
        
        # 3. Define Feature Storage and Hook Setup
        self.dl_feature = None
        self._set_hook()

        # 4. Define Fusion Layers
        self.radiomics_fc = nn.Linear(RADIOMICS_FEATURE_DIM, 256)
        self.radiomics_relu = nn.ReLU()
        self.fusion_fc1 = nn.Linear(512, 256) 
        self.fusion_relu = nn.ReLU()
        self.fusion_fc2 = nn.Linear(256, 2)

    def _forward_hook(self, module, input):
        """Stores the deep learning feature captured right before the final classification."""
        self.dl_feature = input[0]

    def _set_hook(self):
        """Sets the forward pre-hook on the original model's last layer."""
        self.hook = self.model0.fc[2].register_forward_pre_hook(
            lambda module, input: self._forward_hook(module, input)
        )
    
    def forward(self, input_img: torch.Tensor, input_radiomics: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the fusion model.
        """
        
        # 1. Image Feature Extraction
        _ = self.model0(input_img)
        
        if self.dl_feature is None:
            raise RuntimeError("Deep learning feature was not captured by the forward hook.")

        # 2. Radiomics Feature Processing
        radiomics_feature = self.radiomics_relu(self.radiomics_fc(input_radiomics))
        
        # 3. Feature Concatenation
        concat_feature = torch.cat((self.dl_feature, radiomics_feature), dim=1)
        
        # 4. Fusion Classification Head
        outputs = self.fusion_relu(self.fusion_fc1(concat_feature))
        outputs = self.fusion_fc2(outputs)

        self.dl_feature = None
               
        return outputs

if __name__ == '__main__':
    dummy_path = "temp_dummy_us_model.pth"
    torch.save(models.resnet34(weights=None).state_dict(), dummy_path)
    
    try:
        fusion_model = USFeatureFusionModel(pretrained_model_path=dummy_path)
        print("US Feature Fusion Model initialized.")
        dummy_img_input = torch.randn(8, 1, 256, 256) 
        dummy_radiomics_input = torch.randn(8, RADIOMICS_FEATURE_DIM) 
        dummy_output = fusion_model(dummy_img_input, dummy_radiomics_input)
        print(f"\nTest Image input shape: {dummy_img_input.shape}")
        print(f"Test Radiomics input shape: {dummy_radiomics_input.shape}")
        print(f"Test output shape: {dummy_output.shape}") 

    except Exception as e:
        print(f"Test failed: {e}")
        
    finally:
        os.remove(dummy_path)