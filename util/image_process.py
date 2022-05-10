import cv2 as cv
import numpy as np


def adaptive_light_correction(img: str):
    """自适应的光照不均匀校正方法

    Args:
        img (str): 输入图像

    Returns:
        cv.Mat: 校正后的图像
    """
    height = img.shape[0]
    width = img.shape[1]

    HSV_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    V = HSV_img[:, :, 2]

    kernel_size = min(height, width)
    if kernel_size % 2 == 0:
        kernel_size -= 1

    SIGMA1 = 15
    SIGMA2 = 80
    SIGMA3 = 250
    q = np.sqrt(2.0)
    F = np.zeros((height, width, 3), dtype=np.float64)
    F[:, :, 0] = cv.GaussianBlur(V, (kernel_size, kernel_size), SIGMA1 / q)
    F[:, :, 1] = cv.GaussianBlur(V, (kernel_size, kernel_size), SIGMA2 / q)
    F[:, :, 2] = cv.GaussianBlur(V, (kernel_size, kernel_size), SIGMA3 / q)
    F_mean = np.mean(F, axis=2)
    average = np.mean(F_mean)
    # return F_mean - average
    gamma = np.power(0.5, np.divide(np.subtract(average, F_mean), average))
    out = np.power(V/255.0, gamma)*255.0
    HSV_img[:, :, 2] = out
    img = cv.cvtColor(HSV_img, cv.COLOR_HSV2BGR)
    return img
