import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

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