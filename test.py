import os
from os import listdir
from os.path import join, splitext
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize, Compose
from model import Restormer
from PIL import Image
class CustomDatasetTest(data.Dataset):
    def __init__(self, noisy_image_paths, transform=None):
        self.noisy_image_paths = [os.path.join(noisy_image_paths, x) for x in os.listdir(noisy_image_paths)]
        self.transform = transform

    def __len__(self):
        return len(self.noisy_image_paths)

    def __getitem__(self, index):
        noisy_image_path = self.noisy_image_paths[index]
        noisy_image = load_img(self.noisy_image_paths[index])

        if isinstance(noisy_image, np.ndarray):
            noisy_image = Image.fromarray(noisy_image)

        if self.transform:
            noisy_image = self.transform(noisy_image)

        return noisy_image, noisy_image_path


test_transform = Compose([
    ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

model = Restormer()
model.load_state_dict(torch.load('./best_modelll.pth'))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


test_data_path = './test/Input'
output_path = './test/ssssubmissionnnn'

test_dataset = CustomDatasetTest(test_data_path, transform=test_transform)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

if not os.path.exists(output_path):
    os.makedirs(output_path)

for noisy_image, noisy_image_path in test_loader:
    noisy_image = noisy_image.to(device)
    denoised_image = model(noisy_image)

    denoised_image = denoised_image.cpu().squeeze(0)
    denoised_image = (denoised_image * 0.5 + 0.5).clamp(0, 1)
    denoised_image = transforms.ToPILImage()(denoised_image)

    output_filename = noisy_image_path[0]
    denoised_filename = output_path + '/' + output_filename.split('/')[-1][:-4] + '.jpg'
    denoised_image.save(denoised_filename)

    print(f'Saved denoised image: {denoised_filename}')
