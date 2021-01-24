import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from network import DeblurCNN, get_model

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
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

def fit(model, dataloader, epoch):
    """
    Train the neural network

    Args:
        model ([nn.Module]): [DeblurCNN model]
        dataloader ([torch.utils.data]): [Load training dataset]
        epoch ([int]): [Number of epochs to train the neural network]

    Returns:
        [float]: [Training loss]
    """
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