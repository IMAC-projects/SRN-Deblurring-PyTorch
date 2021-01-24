import torch
import cv2
from skimage.metrics import structural_similarity

X = cv2.imread("../dataset/sharp/30_HONOR-8X_S.jpg")
Y = cv2.imread("../dataset/gaussian_blurred/30_HONOR-8X_S.jpg")

ssim_skimage = structural_similarity(X, Y, win_size=11, multichannel=True,sigma=1.5, data_range=1, use_sample_covariance=False, gaussian_weights=True)

print(ssim_skimage)