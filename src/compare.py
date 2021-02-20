import torch
import cv2
import numpy as np
from skimage.metrics import structural_similarity

def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

X = cv2.imread("../outputs/AE_saved_images/sharp0.jpg")
Y = cv2.imread("../outputs/AE_saved_images/val_deblurred499.jpg")

img1 = torch.from_numpy(np.rollaxis(X, 2)).float().unsqueeze(0)/255.0
img2 = torch.from_numpy(np.rollaxis(Y, 2)).float().unsqueeze(0)/255.0

ssim_skimage = structural_similarity(X, Y, win_size=11, multichannel=True,sigma=1.5, data_range=1, use_sample_covariance=False, gaussian_weights=True)

print("SSIM : ")
print(ssim_skimage)
print("PSNR : ")
print(calc_psnr(img1,img2).item())
