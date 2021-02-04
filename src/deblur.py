import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import cv2
import torch
import torchvision
import time
import argparse
from os import scandir
from os.path import isdir, join
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split

from model import get_model
from get_dataset import DeblurDataset, get_train_dataset, get_validation_dataset
from train_validate import util, fit, validate

# helper functions
image_dir = '../outputs/saved_images'
os.makedirs(image_dir, exist_ok=True)


def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders

blur_path = []
sharp_path = []
dirname = fast_scandir("../dataset/GOPRO_Large/train")
sub = 'blur'
sub2 = 'sharp'
for blur in filter(lambda x: sub in x, dirname): 
    blur_path.append(blur)
for sharp in filter(lambda x: sub2 in x, dirname):
    sharp_path.append(sharp)

# constructing the argument parser
def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description='deblur arguments')
    parser.add_argument('-e','--epochs', help='training epoch number', type=int, default=1000)
    parser.add_argument('-b','--batch', help='training batch number', type=int, default=1)
    args = parser.parse_args()
    return args

def split_dataset():
    """
    Split dataset in order to have training and validation set
    75% for training
    25% for validation
    """
    gauss_blur = []
    sharp = []

    for i in range(len(blur_path)):
        _, _, filenames = next(os.walk(blur_path[i]))
        gauss_blur.append(blur_path[i] + "/" + filenames[i])

    for i in range(len(sharp_path)):
        _, _, filenames = next(os.walk(sharp_path[i]))
        sharp.append(sharp_path[i] + "/" + filenames[i])    

    (x_train, x_val, y_train, y_val) = train_test_split(gauss_blur, sharp, test_size=0.25)
    print(f"Train data instances: {len(x_train)}")
    print(f"Validation data instances: {len(x_val)}")
    return (x_train, x_val, y_train, y_val)

def transform_image():
    """
    Transform the image : resize it to 1024x1024
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])
    return transform

def get_scheduler(model):
    criterion, optimizer = util(model)
    # learning rate schedules seek to adjust the learning rate during training by reducing the learning rate according to a pre-defined schedule
    # here, patience is 5 and factor is 0.5
    # if the loss value does not improve for 5 epochs, the new learning rate will be old_learning_rate * 0.5.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
            optimizer,
            mode='min',
            patience=5,
            factor=0.5,
            verbose=True
    )
    return scheduler

def training_validate(epochs,model,train_data, trainloader, val_data, valloader):
    train_loss  = []
    val_loss = []
    start = time.time()
    scheduler = get_scheduler(model)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss = fit(model, trainloader, train_data, epoch)
        val_epoch_loss = validate(model, valloader, val_data, epoch)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        scheduler.step(val_epoch_loss)
    end = time.time()
    print(f"Took {((end-start)/60):.3f} minutes to train")
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../outputs/loss.png')
    plt.show()
    # save the model to disk
    print('Saving model...')
    torch.save(model.state_dict(), '../outputs/model.pth')
    
def main():
    args = parse_args()
    (x_train, x_val, y_train, y_val) = split_dataset()
    transform = transform_image()
    train_data, train_loader = get_train_dataset(x_train,y_train,transform,args.batch)
    val_data, val_loader = get_validation_dataset(x_val,y_val,transform,args.batch)

    training_validate(args.epochs, get_model(),train_data, train_loader, val_data, val_loader)


if __name__ == '__main__':
    main()