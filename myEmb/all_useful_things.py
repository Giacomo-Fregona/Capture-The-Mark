from math import sqrt
from scipy.signal import convolve2d
from scipy.fft import dct, idct
import os
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt
from cv2 import resize
import matplotlib.pyplot as plt
import cv2
import numpy as np


#CODE FOR WPSNR
from scipy.signal import convolve2d
from math import sqrt

def show(image):
    plt.imshow(image,cmap='gray')
    plt.show()

def DCT(image):
    return dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')

def IDCT(image):
    return idct(idct(image, axis=1, norm='ortho'), axis=0, norm='ortho')

def wpsnr(img1, img2):
    img1 = np.float32(img1)/255.0
    img2 = np.float32(img2)/255.0

    difference = img1-img2
    same = not np.any(difference)
    if same is True:
      return 9999999
    csf = np.genfromtxt('csf.csv', delimiter=',')
    ew = convolve2d(difference, np.rot90(csf,2), mode='valid')
    decibels = 20.0*np.log10(1.0/sqrt(np.mean(np.mean(ew**2))))
    return decibels

def similarity(X, X_star):
  # Computes the similarity measure between the original and the new watermarks.
  s = np.sum(np.multiply(X, X_star)) / np.sqrt(np.sum(np.multiply(X_star, X_star)))
  return s



def awgn(img, std, seed):
  mean = 0.0
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

def invert(mark):#function used to get the complementary mark
    for i in range(len(mark)):
        mark[i] = 1 - mark[i]
    return mark

def random_attack(img):
  i = np.random.randint(1,7)
  if i==1:
      attacked = awgn(img, 18.0, np.random.randint(1,100)) # image, standard deviation, seed
      #print("\n\nAwng attack with wpsnr = {}".format(wpsnr(img, attacked)))
  elif i==2:
      attacked = blur(img, [1, 1])
      #print("\n\nBlur attack with wpsnr = {}".format(wpsnr(img, attacked)))
  elif i==3:
      attacked = sharpening(img, 1, 1)
      #print("\n\nSharpening attack with wpsnr = {}".format(wpsnr(img, attacked)))
  elif i==4:
      attacked = median(img, [3, 5])
      #print("\n\nMedian filtering attack with wpsnr = {}".format(wpsnr(img, attacked)))
  elif i==5:
      attacked = resizing(img, 6/14)
      #print("\n\nResizing attack with wpsnr = {}".format(wpsnr(img, attacked)))
  elif i==6:
      attacked = jpeg_compression(img, 12)
      #print("\n\nJpeg compression attack with wpsnr = {}".format(wpsnr(img, attacked)))

  if wpsnr(img, attacked) < 35: return random_attack(img)
  return attacked
  
# Returns the indices of the area (of size area_size) of the array containing the lowest values
def get_lowest_area(arr, area_size):
    min_sum = float("+inf")
    row_idx, col_idx = 0, 0
    for row in range(arr.shape[0]-area_size):
        for col in range(arr.shape[1]-area_size):
            curr_sum =  np.sum(arr[row:row+area_size, col:col+area_size])
            if curr_sum < min_sum:
                row_idx, col_idx = row, col
                min_sum = curr_sum
    # return arr[row_idx:row_idx+area_size, col_idx:col_idx+area_size]
    return row_idx, col_idx

# Attacks an image once, takes the <size> block least affected area of the image and attacks again on it
def attack_least_affected(image, size=64):
    print("Applying JPEG compression...")
    attacked = jpeg_compression(image, 75)

    # We calculate the difference between the original image and the attacked one
    # diff = abs(image - attacked)
    diff = image - attacked

    # We use the previously calculated diff the get the indices of the area with the lowest values, which means the least altered one
    row, col = get_lowest_area(diff, size)
    print("The lowest", size, "pixel area altered by the attack starts at coordinates (", row, col,")")

    # Attacks on the area defined by the previous calculation
    area_to_attack = attacked[row:row+size, col:col+size]
    print("Applying JPEG compression on this area...")
    attacked_area = jpeg_compression(area_to_attack, 75)
    
    # Replacing the area in the attacked image by the newly attacked one
    attacked[row:row+size, col:col+size] = attacked_area
    print("Done. Attacked image is:\n", attacked)
    return attacked

# Attacks an area of an image based on the input indices (row, col) and a size
def attack_area(image, row, col, size):
    area_to_attack = image[row:row+size, col:col+size]
    attacked = jpeg_compression(area_to_attack, 75)
    image[row:row+size, col:col+size] = attacked
    return image

def borders(image):
    border_cut = 10
    area_to_attack = image[:border_cut, :]
    attacked = jpeg_compression(area_to_attack, 1)
    image[:border_cut, :] = attacked

    area_to_attack = image[:, :border_cut]
    attacked = jpeg_compression(area_to_attack, 1)
    image[:, :border_cut] = attacked

    area_to_attack = image[:, 512 - border_cut:]
    attacked = jpeg_compression(area_to_attack, 1)
    image[:, 512 - border_cut:] = attacked

    area_to_attack = image[512 - border_cut:, :]
    attacked = jpeg_compression(area_to_attack, 1)
    image[512 - border_cut:, :] = attacked

    return image