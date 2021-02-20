import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)


class SimpleAE(nn.Module):
    def __init__(self):
        super(SimpleAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(True))
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(64, 32, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=5),
            nn.ReLU(True))
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DeblurCNN(nn.Module):
    """
    Convolutional neural network for deblur

    Args:
        nn.Module([torch.nn.Module]): [Base class for all neural network modules]
    """
    def __init__(self):
        super(DeblurCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

def get_model():
    """
    Get neural network model
    """
    model = DeblurCNN().to(device)
    print(model)
    return model