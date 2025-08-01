# plate_recognition.py
# UNET-based License Plate Recognition System

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from torchvision import transforms as T
from torch import nn
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss
import cv2

# IMPORTS
print("Loading libraries...")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Custom Dataset Class
class Plates_Dataset(Dataset):
    """
    Custom dataset class for loading plate images and their corresponding masks
    """
    def __init__(self, data_path, mask_path, transform=None):
        self.data_path = data_path
        self.mask_path = mask_path
        self.transform = transform
        
        # Get all image files
        self.images = [f for f in os.listdir(data_path) if f.endswith('.jpg')]
        self.images.sort()
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.data_path, img_name)
        mask_path = os.path.join(self.mask_path, img_name)
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

# UNET Model Architecture
class UNET(nn.Module):
    """
    UNET architecture for semantic segmentation
    """
    def __init__(self, channels_in=3, channels=32, num_classes=1):
        super(UNET, self).__init__()
        
        # Encoder (downsampling path)
        self.enc1 = self._make_layer(channels_in, channels, 2)
        self.enc2 = self._make_layer(channels, channels*2, 2)
        self.enc3 = self._make_layer(channels*2, channels*4, 2)
        self.enc4 = self._make_layer(channels*4, channels*8, 2)
        
        # Bottleneck
        self.bottleneck = self._make_layer(channels*8, channels*16, 2)
        
        # Decoder (upsampling path)
        self.up4 = nn.ConvTranspose2d(channels*16, channels*8, kernel_size=2, stride=2)
        self.dec4 = self._make_layer(channels*16, channels*8, 1)
        
        self.up3 = nn.ConvTranspose2d(channels*8, channels*4, kernel_size=2, stride=2)
        self.dec3 = self._make_layer(channels*8, channels*4, 1)
        
        self.up2 = nn.ConvTranspose2d(channels*4, channels*2, kernel_size=2, stride=2)
        self.dec2 = self._make_layer(channels*4, channels*2, 1)
        
        self.up1 = nn.ConvTranspose2d(channels*2, channels, kernel_size=2, stride=2)
        self.dec1 = self._make_layer(channels*2, channels, 1)
        
        # Final output layer
        self.final = nn.Conv2d(channels, num_classes, kernel_size=1)
        
    def _make_layer(self, in_channels, out_channels, num_blocks):
        """Helper function to create a layer with multiple blocks"""
        layers = []
        for _ in range(num_blocks):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Decoder with skip connections
        dec4 = self.up4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.up3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Final output
        output = self.final(dec1)
        return torch.sigmoid(output)

# Training Function
def train(model, train_data, validate_data, optimizer, epochs=100, step_store=10, patience=10, tol_error=1e-2):
    """
    Training function with early stopping
    """
    model.train()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_data):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # Calculate loss (using focal loss for better training)
            loss = sigmoid_focal_loss(output, target, alpha=0.25, gamma=2.0, reduction='mean')
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in validate_data:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = sigmoid_focal_loss(output, target, alpha=0.25, gamma=2.0, reduction='mean')
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_data)
        avg_val_loss = val_loss / len(validate_data)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Print progress
        if epoch % step_store == 0:
            print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss - tol_error:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    return train_losses, val_losses

# Prediction Function
def predict_plate(model, image_path, threshold=0.5):
    """
    Predict plate location in a given image
    """
    model.eval()
    
    # Load and preprocess image
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        prediction = (output > threshold).float()
    
    return prediction.squeeze().cpu().numpy()

# Visualization Function
def visualize_prediction(image_path, prediction, save_path=None):
    """
    Visualize the prediction overlay on the original image
    """
    # Load original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize prediction to match image size
    prediction_resized = cv2.resize(prediction, (image.shape[1], image.shape[0]))
    
    # Create overlay
    overlay = image.copy()
    overlay[prediction_resized > 0.5] = [255, 0, 0]  # Red overlay for detected plate
    
    # Blend images
    alpha = 0.7
    result = cv2.addWeighted(image, alpha, overlay, 1-alpha, 0)
    
    # Plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(prediction_resized, cmap='gray')
    plt.title('Prediction Mask')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(result)
    plt.title('Overlay')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

# Main execution function
def main():
    """
    Main function to run the complete pipeline
    """
    # Configuration
    data_path = "dataset/images"
    mask_path = "dataset/masks"
    batch_size = 8
    learning_rate = 0.001
    epochs = 100
    
    # Data transforms
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    print("Loading dataset...")
    dataset = Plates_Dataset(data_path, mask_path, transform=transform)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    print("Initializing model...")
    model = UNET(channels_in=3, channels=32, num_classes=1).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    print("Starting training...")
    train_losses, val_losses = train(model, train_loader, val_loader, optimizer, epochs=epochs)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    
    # Load best model for prediction
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Example prediction
    if os.path.exists('imgs/test_img.png'):
        print("Making prediction on test image...")
        prediction = predict_plate(model, 'imgs/test_img.png')
        visualize_prediction('imgs/test_img.png', prediction, 'prediction_result.png')
    
    print("Training completed!")

if __name__ == "__main__":
    main()