import os
import numpy as np

from utility import show, wpsnr
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt
from cv2 import resize
import cv2


def awgn(img, std, seed):
  mean = 0.0
  np.random.seed(seed)
  attacked = img + np.random.normal(mean, std, img.shape)
  attacked = np.clip(attacked, 0, 255)
  return attacked

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
  cv2.imwrite('tmp.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), QF])
  attacked = cv2.imread('tmp.jpg', 0)
  os.remove('tmp.jpg')
  return attacked

def random_attack(img):
  i = np.random.randint(1,7)
  #i = 3
  if i==1:
      attacked = awgn(img, 18.0, 123) # image, standard deviation, seed
      #print("\n\nAwng attack with wpsnr = {}".format(wpsnr(img, attacked)))
  elif i==2:
      attacked = blur(img, [1, 1])
      #print("\n\nBlur attack with wpsnr = {}".format(wpsnr(img, attacked)))
  elif i==3:
      attacked = sharpening(img, 0.5, 1) # probabilmente il più pericoloso
      #print("\n\nSharpening attack with wpsnr = {}".format(wpsnr(img, attacked)))
  elif i==4:
      attacked = median(img, [3, 5])
      #print("\n\nMedian filtering attack with wpsnr = {}".format(wpsnr(img, attacked)))
  elif i==5:
      attacked = resizing(img, 6/14)
      #print("\n\nResizing attack with wpsnr = {}".format(wpsnr(img, attacked)))
  elif i==6:
      attacked = jpeg_compression(img, 20)
      #print("\n\nJpeg compression attack with wpsnr = {}".format(wpsnr(img, attacked)))

  if wpsnr(img, attacked) < 35: return random_attack(img)
  return attacked