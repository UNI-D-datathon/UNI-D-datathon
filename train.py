import time
import numpy as np
import os
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import cv2
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.notebook import tqdm
import torch.nn as nn
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from model import Restormer
from PIL import Image
from torchvision.transforms import CenterCrop, Resize

import warnings
warnings.filterwarnings(action='ignore')

CFG = {
    'IMG_SIZE':224,
    'EPOCHS':5,
    'LEARNING_RATE':2e-4,
    'BATCH_SIZE':4,
    'SEED':42
}

class CustomDataset(Dataset):
    def __init__(self, clean_image_paths, noisy_image_paths, transform=None):
        self.clean_image_paths = [os.path.join(clean_image_paths, x) for x in os.listdir(clean_image_paths)]
        self.noisy_image_paths = [os.path.join(noisy_image_paths, x) for x in os.listdir(noisy_image_paths)]
        self.transform = transform
        self.center_crop = CenterCrop(1080)
        self.resize = Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE']))

        self.noisy_clean_pairs = self._create_noisy_clean_pairs()

    def _create_noisy_clean_pairs(self):
        clean_to_noisy = {}
        for clean_path in self.clean_image_paths:
            clean_id = '_'.join(os.path.basename(clean_path).split('_')[:-1])
            clean_to_noisy[clean_id] = clean_path

        noisy_clean_pairs = []
        for noisy_path in self.noisy_image_paths:
            noisy_id = '_'.join(os.path.basename(noisy_path).split('_')[:-1])
            if noisy_id in clean_to_noisy:
                clean_path = clean_to_noisy[noisy_id]
                noisy_clean_pairs.append((noisy_path, clean_path))
            else:
                pass

        return noisy_clean_pairs

    def __len__(self):
        return len(self.noisy_clean_pairs)

    def __getitem__(self, index):
        noisy_image_path, clean_image_path = self.noisy_clean_pairs[index]

        noisy_image = Image.open(noisy_image_path).convert("RGB")
        clean_image = Image.open(clean_image_path).convert("RGB")

        noisy_image = self.center_crop(noisy_image)
        clean_image = self.center_crop(clean_image)
        noisy_image = self.resize(noisy_image)
        clean_image = self.resize(clean_image)

        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

        return noisy_image, clean_image
    
start_time = time.time()

class EarlyStopping:
    def __init__(self, patience=2, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.Inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model, model_path='best_modelcttl.pth'):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            self.counter += 1
            torch.save(model.state_dict(), model_path)
            if self.counter >= self.patience:
                self.early_stop = True

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')

def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

noisy_image_paths = './data/Training/noisy'
clean_image_paths = './data/Training/clean'
val_noisy_image_paths = './data/Validation/noisy'
val_clean_image_paths = './data/Validation/clean'

train_transform = transforms.Compose([
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = CustomDataset(clean_image_paths, noisy_image_paths, transform=train_transform)
val_dataset = CustomDataset(val_clean_image_paths, val_noisy_image_paths, transform=val_transform)

num_cores = os.cpu_count()
train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], num_workers=int(num_cores/2), shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], num_workers=int(num_cores/2), shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Restormer().to(device)

optimizer = optim.AdamW(model.parameters(), lr=CFG['LEARNING_RATE'], weight_decay=1e-4)
criterion = nn.L1Loss()
scaler = GradScaler()
scheduler = CosineAnnealingLR(optimizer, T_max=CFG['EPOCHS'])

total_parameters = count_parameters(model)
print("Total Parameters:", total_parameters)

early_stopping = EarlyStopping(patience=2, min_delta=0.001)

model.train()
model.load_state_dict(torch.load("./best_modelll.pth"), strict=False)
best_loss = 1000

for epoch in range(CFG['EPOCHS']):
    model.train()
    epoch_start_time = time.time()
    mse_running_loss = 0.0

    for noisy_images, clean_images in train_loader:
        noisy_images = noisy_images.to(device)
        clean_images = clean_images.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(noisy_images)
            mse_loss = criterion(outputs, clean_images)

        scaler.scale(mse_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        mse_running_loss += mse_loss.item() * noisy_images.size(0)

    current_lr = scheduler.get_last_lr()[0]
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    minutes = int(epoch_time // 60)
    seconds = int(epoch_time % 60)
    hours = int(minutes // 60)
    minutes = int(minutes % 60)

    mse_epoch_loss = mse_running_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{CFG['EPOCHS']}, MSE Loss: {mse_epoch_loss:.4f}, Lr: {current_lr:.8f}")
    print(f"1epoch 훈련 소요 시간: {hours}시간 {minutes}분 {seconds}초")

    model.eval()
    mse_val_loss = 0.0
    with torch.no_grad():
        for val_noisy_images, val_clean_images in val_loader:
            val_noisy_images = val_noisy_images.to(device)
            val_clean_images = val_clean_images.to(device)

            with autocast():
                val_outputs = model(val_noisy_images)
                val_loss = criterion(val_outputs, val_clean_images)

            mse_val_loss += val_loss.item() * val_noisy_images.size(0)

    mse_val_loss /= len(val_dataset)
    print(f"Validation MSE Loss: {mse_val_loss:.4f}")

    early_stopping(mse_val_loss, model)

    if early_stopping.early_stop:
        print("Early stopping triggered. Training stopped.")
        break

end_time = time.time()

training_time = end_time - start_time
minutes = int(training_time // 60)
seconds = int(training_time % 60)
hours = int(minutes // 60)
minutes = int(minutes % 60)

print(f"훈련 소요 시간: {hours}시간 {minutes}분 {seconds}초")