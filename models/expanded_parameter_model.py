#!/usr/bin/env python3
"""
Enhanced Parameter Predictor Model Definition
ResNet-50 based architecture for mathematical parameter extraction.
"""

import torch
import torch.nn as nn
from torchvision import models


class EnhancedParameterPredictor(nn.Module):
    """ResNet-50 based parameter extraction model for mathematical plots."""
    
    def __init__(self, num_parameters=5):
        """
        Initialize the parameter extraction model.
        
        Args:
            num_parameters: Number of parameters to predict (default: 5)
                           Typically: [amplitude, frequency, phase, param_3, param_4]
        """
        super(EnhancedParameterPredictor, self).__init__()

        # Load pre-trained ResNet-50 backbone
        try:
            resnet = models.resnet50(weights='IMAGENET1K_V1')
        except:
            # Fallback for older torchvision
            resnet = models.resnet50(pretrained=True)
        
        # Remove the final classification layer (avgpool + fc)
        # Keep everything up to and including avgpool for feature extraction
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze early layers (optional - can unfreeze for fine-tuning)
        # for param in list(self.backbone.parameters())[:-20]:
        #     param.requires_grad = False
        
        # Regression head for parameter prediction
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),  # ResNet-50 outputs 2048 features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_parameters)  # Output 5 parameters
        )

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input image tensor (batch_size, 3, 224, 224)
            
        Returns:
            Dictionary with 'parameters' key containing predicted parameter tensor
            Shape: (batch_size, num_parameters)
        """
        # Extract features using ResNet backbone
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 2048)
        
        # Predict parameters through regression head
        parameters = self.fc(x)  # (batch_size, num_parameters)
        
        return {'parameters': parameters}

