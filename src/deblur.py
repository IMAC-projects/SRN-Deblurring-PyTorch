import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split

# helper functions
image_dir = '../outputs/saved_images'
os.makedirs(image_dir, exist_ok=True)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 224, 224)
    save_image(img, name)

# constructing the argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='deblur arguments')
    parser.add_argument('-e','--epoch', help='training epoch number', type=int, default=40)
    parser.add_argument('-b','--batch', help='training batch number', type=int, default=2)
    args = parser.parse_args()
    return args

def split_dataset():
    """
    Split dataset in order to have training and validation set
    75% for training
    25% for validation
    """
    gauss_blur = os.listdir('../dataset/gaussian_blurred')
    gauss_blur.sort()
    sharp = os.listdir('../dataset/sharp')
    sharp.sort()
    x_blur = []
    for i in range(len(gauss_blur)):
        x_blur.append(gauss_blur[i])
    y_sharp = []
    for i in range(len(sharp)):
        y_sharp.append(sharp[i])

    (x_train, x_val, y_train, y_val) = train_test_split(x_blur, y_sharp, test_size=0.25)
    print(f"Train data instances: {len(x_train)}")
    print(f"Validation data instances: {len(x_val)}")
    return (x_train, x_val, y_train, y_val)

def transform_image():
    # define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform
    

def main():
    args = parse_args()
    (x_train, x_val, y_train, y_val) = split_dataset()
    transform = transform_image()

if __name__ == '__main__':
    main()