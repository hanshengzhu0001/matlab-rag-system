#!/usr/bin/env python3
"""
Train ResNet-50 parameter extraction model for mathematical plot analysis.
Generates synthetic dataset, trains model, and saves both model and validation metrics.
"""

import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class EnhancedParameterPredictor(nn.Module):
    """ResNet-50 based parameter extraction model."""
    
    def __init__(self, num_parameters=5):
        super(EnhancedParameterPredictor, self).__init__()
        # Load pre-trained ResNet-50
        try:
            resnet = models.resnet50(weights='IMAGENET1K_V1')
        except:
            # Fallback for older torchvision
            resnet = models.resnet50(pretrained=True)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Regression head for parameter prediction
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_parameters)
        )
        
    def forward(self, x):
        # Extract features using ResNet backbone
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Predict parameters
        parameters = self.fc(x)
        return {'parameters': parameters}


class SyntheticPlotDataset(Dataset):
    """Dataset for synthetic mathematical plots with parameter labels."""
    
    def __init__(self, num_samples=6000, image_size=(224, 224), transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform
        self.data = []
        
        logger.info(f"Generating {num_samples} synthetic plot samples...")
        self._generate_dataset()
        
    def _generate_dataset(self):
        """Generate synthetic mathematical plots with parameters."""
        templates = [
            ('sin', self._generate_sin_plot),
            ('cos', self._generate_cos_plot),
            ('tan', self._generate_tan_plot),
            ('sin_cos_sum', self._generate_sin_cos_sum_plot),
            ('polynomial', self._generate_polynomial_plot),
            ('exponential', self._generate_exponential_plot),
        ]
        
        samples_per_template = self.num_samples // len(templates)
        
        for template_name, generator_func in templates:
            for i in range(samples_per_template):
                try:
                    image, params = generator_func()
                    self.data.append({
                        'image': image,
                        'parameters': params,
                        'template': template_name
                    })
                except Exception as e:
                    logger.warning(f"Failed to generate {template_name} sample {i}: {e}")
        
        # Fill remaining samples with random templates
        remaining = self.num_samples - len(self.data)
        for i in range(remaining):
            template_name, generator_func = templates[np.random.randint(0, len(templates))]
            try:
                image, params = generator_func()
                self.data.append({
                    'image': image,
                    'parameters': params,
                    'template': template_name
                })
            except Exception as e:
                logger.warning(f"Failed to generate remaining sample: {e}")
        
        logger.info(f"Generated {len(self.data)} samples successfully")
    
    def _generate_sin_plot(self):
        """Generate sin(x) plot with parameters: [amplitude, frequency, phase, 0, 0]"""
        A = np.random.uniform(0.5, 2.0)  # Amplitude
        freq = np.random.uniform(0.5, 3.0)  # Frequency
        phase = np.random.uniform(0, 2*math.pi)  # Phase
        
        img = Image.new('RGB', self.image_size, 'white')
        draw = ImageDraw.Draw(img)
        
        width, height = self.image_size
        x_vals = np.linspace(0, 4*math.pi, width)
        y_vals = [A * math.sin(freq * x + phase) for x in x_vals]
        
        # Normalize to image coordinates
        y_min, y_max = min(y_vals), max(y_vals)
        y_range = y_max - y_min if y_max > y_min else 1
        y_coords = [int(height/2 - (y - (y_min + y_max)/2) * (height*0.4) / y_range) for y in y_vals]
        
        # Draw curve
        for i in range(len(x_vals)-1):
            draw.line([(i, y_coords[i]), (i+1, y_coords[i+1])], fill='black', width=2)
        
        params = np.array([A, freq, phase, 0.0, 0.0], dtype=np.float32)
        return img, params
    
    def _generate_cos_plot(self):
        """Generate cos(x) plot"""
        A = np.random.uniform(0.5, 2.0)
        freq = np.random.uniform(0.5, 3.0)
        phase = np.random.uniform(0, 2*math.pi)
        
        img = Image.new('RGB', self.image_size, 'white')
        draw = ImageDraw.Draw(img)
        
        width, height = self.image_size
        x_vals = np.linspace(0, 4*math.pi, width)
        y_vals = [A * math.cos(freq * x + phase) for x in x_vals]
        
        y_min, y_max = min(y_vals), max(y_vals)
        y_range = y_max - y_min if y_max > y_min else 1
        y_coords = [int(height/2 - (y - (y_min + y_max)/2) * (height*0.4) / y_range) for y in y_vals]
        
        for i in range(len(x_vals)-1):
            draw.line([(i, y_coords[i]), (i+1, y_coords[i+1])], fill='black', width=2)
        
        params = np.array([A, freq, phase, 0.0, 0.0], dtype=np.float32)
        return img, params
    
    def _generate_tan_plot(self):
        """Generate tan(x) plot"""
        A = np.random.uniform(0.5, 2.0)
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, math.pi/2)
        
        img = Image.new('RGB', self.image_size, 'white')
        draw = ImageDraw.Draw(img)
        
        width, height = self.image_size
        x_vals = np.linspace(-math.pi/2 + 0.1, math.pi/2 - 0.1, width)
        y_vals = [A * math.tan(freq * x + phase) for x in x_vals]
        
        # Clip extreme values
        y_vals = [max(-10, min(10, y)) for y in y_vals]
        
        y_min, y_max = min(y_vals), max(y_vals)
        y_range = y_max - y_min if y_max > y_min else 1
        y_coords = [int(height/2 - (y - (y_min + y_max)/2) * (height*0.4) / y_range) for y in y_vals]
        
        for i in range(len(x_vals)-1):
            if abs(y_coords[i+1] - y_coords[i]) < height:  # Skip discontinuities
                draw.line([(i, y_coords[i]), (i+1, y_coords[i+1])], fill='black', width=2)
        
        params = np.array([A, freq, phase, 0.0, 0.0], dtype=np.float32)
        return img, params
    
    def _generate_sin_cos_sum_plot(self):
        """Generate sin(x) + cos(x) plot"""
        A1 = np.random.uniform(0.5, 2.0)  # sin amplitude
        A2 = np.random.uniform(0.5, 2.0)  # cos amplitude
        freq = np.random.uniform(0.5, 3.0)
        
        img = Image.new('RGB', self.image_size, 'white')
        draw = ImageDraw.Draw(img)
        
        width, height = self.image_size
        x_vals = np.linspace(0, 4*math.pi, width)
        y_vals = [A1 * math.sin(freq * x) + A2 * math.cos(freq * x) for x in x_vals]
        
        y_min, y_max = min(y_vals), max(y_vals)
        y_range = y_max - y_min if y_max > y_min else 1
        y_coords = [int(height/2 - (y - (y_min + y_max)/2) * (height*0.4) / y_range) for y in y_vals]
        
        for i in range(len(x_vals)-1):
            draw.line([(i, y_coords[i]), (i+1, y_coords[i+1])], fill='black', width=2)
        
        params = np.array([A1, freq, A2, 0.0, 0.0], dtype=np.float32)
        return img, params
    
    def _generate_polynomial_plot(self):
        """Generate polynomial plot"""
        a = np.random.uniform(-2.0, 2.0)
        b = np.random.uniform(-1.0, 1.0)
        c = np.random.uniform(-0.5, 0.5)
        
        img = Image.new('RGB', self.image_size, 'white')
        draw = ImageDraw.Draw(img)
        
        width, height = self.image_size
        x_vals = np.linspace(-5, 5, width)
        y_vals = [a*x**2 + b*x + c for x in x_vals]
        
        y_min, y_max = min(y_vals), max(y_vals)
        y_range = y_max - y_min if y_max > y_min else 1
        y_coords = [int(height/2 - (y - (y_min + y_max)/2) * (height*0.4) / y_range) for y in y_vals]
        
        for i in range(len(x_vals)-1):
            draw.line([(i, y_coords[i]), (i+1, y_coords[i+1])], fill='black', width=2)
        
        params = np.array([a, b, c, 0.0, 0.0], dtype=np.float32)
        return img, params
    
    def _generate_exponential_plot(self):
        """Generate exponential plot"""
        A = np.random.uniform(0.5, 2.0)
        k = np.random.uniform(-1.0, 1.0)
        
        img = Image.new('RGB', self.image_size, 'white')
        draw = ImageDraw.Draw(img)
        
        width, height = self.image_size
        x_vals = np.linspace(0, 5, width)
        y_vals = [A * math.exp(k * x) for x in x_vals]
        
        y_min, y_max = min(y_vals), max(y_vals)
        y_range = y_max - y_min if y_max > y_min else 1
        y_coords = [int(height - (y - y_min) * (height*0.8) / y_range) for y in y_vals]
        
        for i in range(len(x_vals)-1):
            draw.line([(i, y_coords[i]), (i+1, y_coords[i+1])], fill='black', width=2)
        
        params = np.array([A, k, 0.0, 0.0, 0.0], dtype=np.float32)
        return img, params
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image']
        params = item['parameters']
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform
            transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            image = transform(image)
        
        return image, torch.FloatTensor(params)


def train_model(num_epochs=10, batch_size=32, learning_rate=0.001, num_samples=6000):
    """Train the parameter extraction model."""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    full_dataset = SyntheticPlotDataset(num_samples=num_samples, transform=transform)
    
    # Split into train/validation (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    model = EnhancedParameterPredictor(num_parameters=5).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'epochs': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    logger.info("Starting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs['parameters'], targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_rmse = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                loss = criterion(outputs['parameters'], targets)
                val_loss += loss.item()
                
                # Calculate RMSE
                mse_per_sample = torch.mean((outputs['parameters'] - targets) ** 2, dim=1)
                rmse_per_sample = torch.sqrt(mse_per_sample)
                val_rmse += torch.mean(rmse_per_sample).item()
                
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        avg_val_rmse = val_rmse / val_batches
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            logger.info(f"✅ New best model! Validation Loss: {avg_val_loss:.6f}")
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_rmse'].append(avg_val_rmse)
        history['epochs'].append(epoch + 1)
        
        logger.info(f"Epoch [{epoch+1}/{num_epochs}]")
        logger.info(f"  Train Loss: {avg_train_loss:.6f}")
        logger.info(f"  Val Loss: {avg_val_loss:.6f}")
        logger.info(f"  Val RMSE: {avg_val_rmse:.6f}")
        logger.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")
    
    # Save best model
    model_dir = Path(__file__).parent
    model_path = model_dir / "enhanced_parameter_predictor_best.pth"
    torch.save(best_model_state, model_path)
    logger.info(f"✅ Saved best model to {model_path}")
    
    # Save training history and metrics
    metrics_path = model_dir / "training_metrics.json"
    final_metrics = {
        'best_val_loss': float(best_val_loss),
        'best_val_rmse': float(history['val_rmse'][-1]),
        'final_train_loss': float(history['train_loss'][-1]),
        'final_val_loss': float(history['val_loss'][-1]),
        'final_val_rmse': float(history['val_rmse'][-1]),
        'total_epochs': num_epochs,
        'training_samples': len(train_dataset),
        'validation_samples': len(val_dataset),
        'history': {
            'epochs': history['epochs'],
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'val_rmse': [float(x) for x in history['val_rmse']]
        }
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    logger.info(f"✅ Saved training metrics to {metrics_path}")
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Best Validation Loss (MSE): {best_val_loss:.6f}")
    logger.info(f"Best Validation RMSE: {history['val_rmse'][-1]:.6f}")
    logger.info(f"Final Training Loss: {history['train_loss'][-1]:.6f}")
    logger.info(f"Final Validation Loss: {history['val_loss'][-1]:.6f}")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Metrics saved to: {metrics_path}")
    
    return model, final_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train parameter extraction model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--samples', type=int, default=6000, help='Number of synthetic samples')
    
    args = parser.parse_args()
    
    train_model(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_samples=args.samples
    )
