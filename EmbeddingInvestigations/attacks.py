import os
import numpy as np
from utility import show, wpsnr


#this seed was set just to make you obtain the same result
np.random.seed(3)
def awgn(img, std, seed):
  mean = 0.0   # some constant
  #np.random.seed(seed)
  attacked = img + np.random.normal(mean, std, img.shape)
  attacked = np.clip(attacked, 0, 255)
  return attacked

def blur(img, sigma):
  from scipy.ndimage.filters import gaussian_filter
  attacked = gaussian_filter(img, sigma)
  return attacked

def sharpening(img, sigma, alpha):
  import scipy
  from scipy.ndimage import gaussian_filter
  import matplotlib.pyplot as plt

  #print(img/255)
  filter_blurred_f = gaussian_filter(img, sigma)

  attacked = img + alpha * (img - filter_blurred_f)
  return attacked

def median(img, kernel_size):
  from scipy.signal import medfilt
  attacked = medfilt(img, kernel_size)
  return attacked

def resizing(img, scale):
  from skimage.transform import rescale
  x, y = img.shape
  attacked = rescale(img, scale)
  attacked = rescale(attacked, 1/scale)
  attacked = attacked[:x, :y]
  return attacked

def jpeg_compression(img, QF):
  from PIL import Image
  img = Image.fromarray(img)
  img.save('tmp.jpg',"JPEG", quality=QF)
  attacked = Image.open('tmp.jpg')
  attacked = np.asarray(attacked,dtype=np.uint8)
  os.remove('tmp.jpg')

  return attacked

def random_attack(img):
  i = np.random.randint(1,7)
  if i==1:
      attacked = awgn(img, 5.0, 123)
      #print("\n\nAwng attack with wpsnr = {}".format(wpsnr(img, attacked)))
  elif i==2:
      attacked = blur(img, [3, 2])
      #print("\n\nBlur attack with wpsnr = {}".format(wpsnr(img, attacked)))
  elif i==3:
      attacked = sharpening(img, 1, 1)
      #print("\n\nSharpening attack with wpsnr = {}".format(wpsnr(img, attacked)))
  elif i==4:
      attacked = median(img, [3, 5])
      #print("\n\nMedian filtering attack with wpsnr = {}".format(wpsnr(img, attacked)))
  elif i==5:
      attacked = resizing(img, 4)
      #print("\n\nResizing attack with wpsnr = {}".format(wpsnr(img, attacked)))
  elif i==6:
      attacked = jpeg_compression(img, 75)
      #print("\n\nJpeg compression attack with wpsnr = {}".format(wpsnr(img, attacked)))

  if wpsnr(img, attacked) < 35: return random_attack(img)
  return attacked