import cv2
from torch.utils.data import Dataset, DataLoader

class DeblurDataset(Dataset):
    def __init__(self, blur_paths, sharp_paths=None, transforms=None):
        self.X = blur_paths
        self.y = sharp_paths
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        blur_image = cv2.imread(self.X[i])
        
        if self.transforms:
            blur_image = self.transforms(blur_image)
            
        if self.y is not None:
            sharp_image = cv2.imread(self.y[i])
            sharp_image = self.transforms(sharp_image)
            return (blur_image, sharp_image)
        else:
            return blur_image

def get_train_dataset(x_train, y_train, transform, batch_size):
    train_data = DeblurDataset(x_train, y_train, transform)    
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_data,trainloader

def get_validation_dataset(x_val,y_val, transform, batch_size):
    val_data = DeblurDataset(x_val, y_val, transform)
    valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return val_data,valloader