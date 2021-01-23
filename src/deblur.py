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

def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 224, 224)
    save_image(img, name)

# constructing the argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='deblur arguments')
    parser.add_argument('-e','--epoch', help='training epoch number', type=int, default=40)
    args = parser.parse_args()
    return args

# helper functions
image_dir = '../outputs/saved_images'
os.makedirs(image_dir, exist_ok=True)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
batch_size = 2

def main():
    args = parse_args()

if __name__ == '__main__':
    main()