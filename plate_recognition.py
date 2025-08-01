# Plate Recognition in Images using Unet as Neural Network model
# This script is intended to be run and trained on a Linux system with an AMD GPU using ROCm.
# Make sure your environment is properly configured for ROCm and PyTorch GPU support.

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

# Choose the device, will use GPU if available (AMD ROCm compatible)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else "No CUDA")
print("Number of GPUs:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU detected:")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
else:
    print("No GPU detected. Possible issues:")
    print("1. ROCm PyTorch not properly installed")
    print("2. Environment variables not set correctly")
    print("3. GPU not supported by current ROCm version")

my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {my_device}")

# Force GPU usage if available (comment this line if you want to test on CPU)
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Use first GPU
    print(f"Current GPU: {torch.cuda.current_device()}")
else:
    print("WARNING: Running on CPU - training will be very slow!")

# FILE PATHS
# Use relative paths for Linux compatibility
PATH_DATA = 'dataset/images/'
PATH_MASK = 'dataset/masks/'

# DATASET
class Plates_Dataset(Dataset):
    def __init__(self, data_path, mask_path, transform_data, transform_mask):
        self.data_path = data_path
        self.mask_path = mask_path
        self.transform_data = transform_data
        self.transform_mask = transform_mask
        self.images = sorted(os.listdir(self.data_path))
        self.masks = sorted(os.listdir(self.mask_path))
    def __len__(self):
        return len(self.masks)
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.images[idx])
        image = Image.open(img_path)
        mask_path = os.path.join(self.mask_path, self.masks[idx])
        mask = Image.open(mask_path)
        if self.transform_data is not None:
            image = self.transform_data(image)
        if self.transform_mask is not None:
            mask = self.transform_mask(mask)
        mask = (torch.sum(mask, dim=0) / 3).unsqueeze(0)
        # image is returned in 3 channels (rgb), mask in 1 channel (grayscale)
        return image, mask

# transforms
transform = T.Compose([T.Resize([224, 224]), T.ToTensor()])

# call of the dataset class
full_dataset = Plates_Dataset(PATH_DATA, PATH_MASK, transform, transform)

# CONSTANTS
DATASET_SIZE = len(full_dataset)
BATCH_SIZE = 32
TRAIN_SIZE = int(0.80 * DATASET_SIZE)
TEST_SIZE = DATASET_SIZE - TRAIN_SIZE
VALID_SIZE = int(0.1 * TRAIN_SIZE)

# split the complete train_dataset in two tensors:
t_dataset, test_dataset = random_split(full_dataset, [TRAIN_SIZE, TEST_SIZE])
train_dataset, valid_dataset = random_split(t_dataset, [TRAIN_SIZE - VALID_SIZE, VALID_SIZE])

print('Length of train dataset: ', len(train_dataset))
print('Length of validation dataset: ', len(valid_dataset))
print('Length of test dataset: ', len(test_dataset))
print('Total of files in the dataset: ', len(full_dataset))

# DATALOADER
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validate_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

# PLOT THE BATCH
def plot_batch(batch_size, imgs, masks):
    plt.figure(figsize=(20, 10))
    for i in range(batch_size):
        plt.subplot(4, 8, i + 1)
        img = imgs[i, ...].permute(1, 2, 0).numpy()
        mask = masks[i, ...].permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.imshow(mask, cmap='gray', alpha=0.75)
        plt.axis('Off')
    plt.tight_layout()
    plt.show()

# plot some examples
imgs, masks = next(iter(train_loader))
print(imgs.shape, masks.shape)
plot_batch(BATCH_SIZE, imgs, masks)

class Conv_3_k(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        return self.conv1(x)

class Double_Conv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.double_conv = nn.Sequential(
            Conv_3_k(channels_in, channels_out),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
            Conv_3_k(channels_out, channels_out),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.double_conv(x)

class Down_Conv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2, 2),
            Double_Conv(channels_in, channels_out)
        )
    def forward(self, x):
        return self.encoder(x)

class Up_Conv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic'),
            nn.Conv2d(channels_in, channels_in // 2, kernel_size=1, stride=1)
        )
        self.decoder = Double_Conv(channels_in, channels_out)
    def forward(self, x1, x2):
        x1 = self.upsample_layer(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.decoder(x)

# UNET implementation
class UNET(nn.Module):
    def __init__(self, channels_in, channels, num_classes):
        super().__init__()
        self.first_conv = Double_Conv(channels_in, channels)
        self.down_conv1 = Down_Conv(channels, 2 * channels)
        self.down_conv2 = Down_Conv(2 * channels, 4 * channels)
        self.down_conv3 = Down_Conv(4 * channels, 8 * channels)
        self.middle_conv = Down_Conv(8 * channels, 16 * channels)
        self.up_conv1 = Up_Conv(16 * channels, 8 * channels)
        self.up_conv2 = Up_Conv(8 * channels, 4 * channels)
        self.up_conv3 = Up_Conv(4 * channels, 2 * channels)
        self.up_conv4 = Up_Conv(2 * channels, channels)
        self.last_conv = nn.Conv2d(channels, num_classes, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x1 = self.first_conv(x)
        x2 = self.down_conv1(x1)
        x3 = self.down_conv2(x2)
        x4 = self.down_conv3(x3)
        x5 = self.middle_conv(x4)
        u1 = self.up_conv1(x5, x4)
        u2 = self.up_conv2(u1, x3)
        u3 = self.up_conv3(u2, x2)
        u4 = self.up_conv4(u3, x1)
        return self.sigmoid(self.last_conv(u4))

# Metrics function
def metrics(model, data):
    intersection = 0
    denom = 0
    union = 0
    loss_acum = 0
    model.to(device=my_device)
    with torch.no_grad():
        for x, y in data:
            x = x.to(device=my_device, dtype=torch.float32)
            y = y.to(device=my_device, dtype=torch.float32).squeeze(1)
            y_pred = model(x).squeeze(1)
            loss = F.binary_cross_entropy(y_pred, y).to(device=my_device)
            loss_acum += loss.item()
            intersection += (y_pred * y).sum()
            denom += (y_pred + y).sum()
            union += (y_pred + y - y_pred * y).sum()
        dice = 2 * intersection / (denom + 1e-8)
        iou = (intersection) / (union + 1e-5)
    return dice, iou, loss_acum

# Train function
def train(model, train_data, validate_data, optimizer, epochs=100, step_store=10, pacient=10, tol_error=1e-2):
    acum_v = []
    model = model.to(device=my_device)
    iou_v = []
    dice_v = []
    model.train()
    for epoch in range(epochs):
        acum = 0
        pacient_step = 0
        for batch, (x, y) in enumerate(train_data, start=1):
            x = x.to(device=my_device, dtype=torch.float32)
            y = y.to(device=my_device, dtype=torch.float32)
            y_pred = model(x)
            optimizer.zero_grad()
            loss = F.binary_cross_entropy(y_pred, y).to(device=my_device)
            acum += loss.item()
            loss.backward()
            optimizer.step()
            if batch % step_store == 0:
                dice, iou, loss_val = metrics(model, validate_data)
                iou_v.append(iou)
                dice_v.append(dice)
                print(f' dice: {dice}, iou: {iou}')
                if loss_val < tol_error:
                    if pacient_step < pacient:
                        pacient_step += 1
                    else:
                        break
        print('epoch ', epoch, ' acumulated: ', acum)
        acum_v.append(acum)
    # plt.plot(acum_v)
    # plt.show()
    return iou_v, dice_v

# Define the model and optimizer
model = UNET(channels_in=3, channels=32, num_classes=1)
epochs = 20
optimizer_unet = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.95)

# Train the model
iou_v, dice_v = train(model, train_loader, validate_loader, optimizer_unet, epochs)

torch.save(model, 'model8c_5e_ce.pt')
torch.save(model.state_dict(), 'model16c_50e_bcelog_state_dict.pt')

# Plot metrics
x = BATCH_SIZE * np.arange(0, len(iou_v))
plt.plot(x, torch.Tensor(iou_v), label='IOU')
plt.plot(x, torch.Tensor(dice_v), label='DICE')
plt.xlabel('')
plt.ylabel('')
plt.title('Metrics for training: IOU, DICE')
plt.legend(loc="upper left")
plt.show()

# TEST
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Plot predictions
imgs_test, masks_test = next(iter(test_loader))
model = model.to(device=my_device)
imgs_test = imgs_test.cuda()
with torch.no_grad():
    scores = model(imgs_test)
    preds = scores[:, 0]
imgs_test = imgs_test.cpu()
preds = preds.cpu()
print(preds.shape)
plot_batch(BATCH_SIZE, imgs_test, preds.unsqueeze(1))

def test(model, data):
    dice, iou = metrics(model, data)
    return dice, iou

iou_v_test, dice_v_test = test(model, test_loader)

# Plot test metrics
x = BATCH_SIZE * np.arange(0, len(iou_v))
plt.plot(x, torch.Tensor(iou_v), label='IOU')
plt.plot(x, torch.Tensor(dice_v), label='DICE')
plt.xlabel('')
plt.ylabel('')
plt.title('Metrics for TEST: IOU, DICE')
plt.legend(loc="upper left")
plt.show()
