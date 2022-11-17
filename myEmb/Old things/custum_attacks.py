import os
import numpy as np
from PIL.Image import Image

from utility import show, wpsnr
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt
from cv2 import resize
import cv2
from attacks import *
from main import embedding, detection


#this seed was set just to make you obtain the same result
#np.random.seed(3)
def awgn(img, std, seed):
    mean = 0.0
    np.random.seed(seed)
    attacked = img + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked

"""
def adaptive_awgn(img,seed,max=30,precision=10):
    std = 0
    attacked = np.zeros_like(img)
    while (std <= max and wpsnr(img, attacked)<35):
        std += step
        attacked = awgn(img,std,seed)
    if (std == max):
        return attacked
    while detection(img, awgn(img,max,seed)):
        
    for i in range(precision):
        if wpsnr(img, awgn(img,(stdUp+stdDown)/2,seed))>35:
            stdUp = (stdUp+stdDown)/2
        else:
            stdDown = (stdUp + stdDown) / 2
    
    
    stdUp = std
    stdDown = std-step
    for i in range(precision):
        if wpsnr(img, awgn(img,(stdUp+stdDown)/2,seed))>35:
            stdUp = (stdUp+stdDown)/2
        else:
            stdDown = (stdUp + stdDown) / 2
    return awgn(img,stdUp,seed)
    """
lena = Image.open('lena.bmp')
lena = np.asarray(Image.open('lena.bmp'), dtype=np.uint8)

mark, wd = embedding(lena)

w_ex = detection(lena, wd)

print('wPSNR for attlena = %.2fdB' % wpsnr(lena, wd))









def blur(img, sigma):
    attacked = gaussian_filter(img, sigma)
    return attacked


def sharpening(img, sigma, alpha):
    filter_blurred_f = gaussian_filter(img, sigma)
    attacked = img + alpha * (img - filter_blurred_f)
    return attacked

def median(img, kernel_size):
    attacked = medfilt(img, kernel_size)
    return attacked

def resizing(img, scale):
    x, y = img.shape
    _x = int(x*scale)
    _y = int(y*scale)
    attacked = resize(img, (_x, _y))
    attacked = resize(attacked, (x, y))
    return attacked

def jpeg_compression(img, QF):
    cv2.imwrite('tmp.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    attacked = cv2.imread('tmp.jpg', 0)
    os.remove('tmp.jpg')
    return attacked