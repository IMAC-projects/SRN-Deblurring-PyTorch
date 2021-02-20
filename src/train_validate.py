import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image

from model import DeblurCNN, get_model

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def util(model):
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters()}], lr=0.001)
    '''
    optimizer = optim.Adam([
        {'params': model.encoder.parameters()},
        {'params': model.decoder.parameters()}], lr=0.001)
    '''
    return criterion, optimizer

def fit(model, dataloader, train_data, epoch):
    """
    Train the model

    Args:
        model ([nn.Module]): [DeblurCNN model]
        dataloader ([torch.utils.data]): [Load training dataset]
        epoch ([int]): [Number of epochs to train the model]

    Returns:
        [float]: [Training loss]
    """
    criterion, optimizer = util(model)
    
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        blur_image = data[0]
        sharp_image = data[1]
        blur_image = blur_image.to(device)
        sharp_image = sharp_image.to(device)
        optimizer.zero_grad()
        outputs = model(blur_image)
        loss = criterion(outputs, sharp_image)
        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step()
        running_loss += loss.item()
    
    train_loss = running_loss/len(dataloader.dataset)
    print(f"Train Loss: {train_loss:.5f}")
    
    return train_loss

def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 1024, 1024)
    save_image(img, name)

def validate(model, dataloader, val_data, epoch):
    """
    Neural model validation
    Args:
        model ([nn.Module]): [DeblurCNN model]
        dataloader ([torch.utils.data]): [Load validation dataset]
        epoch ([int]): [Number of epochs to train the neural model]
    Returns:
        [float]: [Validation loss]
    """
    criterion, optimizer = util(model)
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            blur_image = data[0]
            sharp_image = data[1]
            blur_image = blur_image.to(device)
            sharp_image = sharp_image.to(device)
            outputs = model(blur_image)
            loss = criterion(outputs, sharp_image)
            running_loss += loss.item()
            
        val_loss = running_loss/len(dataloader.dataset)
        print(f"Val Loss: {val_loss:.5f}")

        save_decoded_image(outputs.cpu().data, name=f"../outputs/saved_images/val_deblurred{epoch}.jpg")
        
        return val_loss