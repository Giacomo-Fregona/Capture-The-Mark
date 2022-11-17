import os
from utility import wpsnr
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt
from cv2 import resize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import cv2
import random
import numpy as np

#CODE FOR WPSNR
from scipy.signal import convolve2d
from math import sqrt
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
  norm = np.sum(np.multiply(X_star, X_star))
  if norm < 10**(-12):
      return 0
  s = np.sum(np.multiply(X, X_star)) / np.sqrt(norm)
  return s


def compute_confidence(d):#implements a case defined linear function that we use to extimate how much we can trust the bit we got
    if d > 2.5: return 0
    elif d<0.7: return 1
    elif d < 1: return 0.7
    elif d < 1.5: return 0.4
    else: return 0.2

def ROCdetection(input1, input2, input3):# detection of the code with only difference that the similarity is returned

    #opening images
    original = cv2.imread(input1, 0)
    watermarked = cv2.imread(input2, 0)
    attacked = cv2.imread(input3, 0)

    #choosing parameters
    blocksize = 8  # it could be a divisor of 480 = 5 * 3 * 2^5, better if a multiple of 8
    n_blocks = 512 // blocksize
    mark_size = 1024
    mark_repetitions = 1
    key = 1090
    random.seed(key)
    position_list = [(blocksize * i, blocksize * j, 2) for i in range(n_blocks) for j in range(n_blocks)]
    random.shuffle(position_list)

    mark = np.load("pixel.npy")

    # retrieving the watermark

    def is_on_border(pos):
        return not ((pos[0] > 12) and (pos[0] < 500) and (pos[1] > 12) and (pos[1] < 500))

    def swap(position_list, i, j):
        temp = position_list[i]
        position_list[i] = position_list[j]
        position_list[j] = temp
        return position_list

    zeros_on_border = []
    for i in range(2500):
        if i >= 1024 or mark[i] == 0:
            pos = position_list[i]
            if is_on_border(pos):
                zeros_on_border.append(i)
    random.shuffle(zeros_on_border)
    print(len(zeros_on_border))

    ones_in_middle = []
    for i in range(1024):
        if mark[i] == 1:
            pos = position_list[i]
            if not is_on_border(pos):
                ones_in_middle.append(i)
    random.shuffle(ones_in_middle)
    print(len(ones_in_middle))

    counter = 0
    for i in range(min(len(ones_in_middle), len(zeros_on_border))):
        position_list = swap(position_list, ones_in_middle[i], zeros_on_border[i])
        counter += 1
    # print("number of swaps = {}".format(counter))

    counter = 0
    for i in range(538):
        if (not is_on_border(position_list[i])) and mark[i] == 1:
            position_list[i] = (position_list[i][0], position_list[i][1], 1)
            counter += 1
    # print("counter = {}".format(counter))
    redundant_mark = np.zeros(mark_repetitions * mark_size)  # bits of the watermark multiple copies extracted

    for k in range(mark_repetitions * mark_size):
        i, j, alpha = position_list[k]
        block_watermarked = watermarked[i:i + blocksize, j:j + blocksize].copy()
        block_image = original[i:i + blocksize, j:j + blocksize].copy()
        block_watermarked = block_watermarked.astype(
            'float64')  # going to float excludes overflow problems in the lines below
        block_image = block_image.astype('float64')

        # Embedding strategy: we sum all the pixel differences between watermarked and original image (score) and
        # divide by the nuber of values we should have added/subtracted if there was a 1 on the mark (divisor)

        divisor = 0
        if block_image.mean() < 128:  # In that case we should look for added alpha
            score = (block_watermarked - block_image).sum()
            for ii in range(blocksize):
                for jj in range(blocksize):
                    divisor += min(alpha, 255 - block_image[ii, jj])
        else:  # In that case we should look for subtracted alpha
            score = (block_image - block_watermarked).sum()
            for ii in range(blocksize):
                for jj in range(blocksize):
                    divisor += min(alpha, block_image[ii, jj])
        redundant_mark[k] = score / divisor

    # calculating and returning the mean value of the bits that we got from different copies
    for i in range(mark_size):
        for j in range(1, mark_repetitions):
            redundant_mark[i] += redundant_mark[j * mark_size + i]
    our_mark = (redundant_mark / mark_repetitions)[:mark_size]

    redundant_mark = np.zeros(mark_repetitions * mark_size)  # bits of the watermark multiple copies extracted

    for k in range(mark_repetitions * mark_size):
        i, j, alpha = position_list[k]
        block_attacked = attacked[i:i + blocksize, j:j + blocksize].copy()
        block_image = original[i:i + blocksize, j:j + blocksize].copy()
        block_attacked = block_attacked.astype(
            'float64')  # going to float excludes overflow problems in the lines below
        block_image = block_image.astype('float64')

        # How to design a strategy with the new embedding? We sum all the pixel differences between watermarked and original image (score) and
        # divide by the nuber of values we should have added/subtracted if there was a 1 on the mark (divisor)

        divisor = 0
        if block_image.mean() < 128:  # In that case we should look for added alpha
            score = (block_attacked - block_image).sum()
            for ii in range(blocksize):
                for jj in range(blocksize):
                    divisor += min(alpha, 255 - block_image[ii, jj])
        else:  # In that case we should look for subtracted alpha
            score = (block_image - block_attacked).sum()
            for ii in range(blocksize):
                for jj in range(blocksize):
                    divisor += min(alpha, block_image[ii, jj])
        redundant_mark[k] = score / divisor




    # calculating and returning the mean value of the bits that we got from different copies
    mark_to_return = np.zeros(mark_size)
    for i in range(mark_size):
        retrieved_bits = []
        confidence = []
        for j in range(mark_repetitions):
            bit = float(redundant_mark[j * mark_size + i])
            retrieved_bits.append(bit)
            confidence.append(compute_confidence(abs(bit - 0.5)))
        #print("we got {} while the right bit is: {}\n".format(retrieved_bits, our_mark[i]))
        if sum(confidence) < 10**(-12):
            mark_to_return[i] = 0.5
        else:
            retrieved_bits = np.clip(retrieved_bits, 0, 1) # excluding values more than 1 or less than 0
            mark_to_return[i] = sum([retrieved_bits[i]*confidence[i] for i in range(len(retrieved_bits))])/sum(confidence)
        #print("we got {} so we return {} while the right bit is: {}\n".format(retrieved_bits,mark_to_return[i],our_mark[i]))
    return mark_to_return







blocksize = 8  # it could be a divisor of 480 = 5 * 3 * 2^5, better if a multiple of 8
n_blocks = 512 // blocksize
mark_size = 1024
mark_repetitions = 2

from embedding_to_copy import embedding

#np.random.seed(239487)

def awgn(img, std, seed):
  mean = 0.0
  #np.random.seed(seed)
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


def show(image):
    plt.imshow(image,cmap='gray')
    plt.show()

def random_attack(img):
  i = np.random.randint(2,7)
  #i= 6
  if i==1:
      attacked = awgn(img, 19.0, 123) # image, standard deviation, seed
      print("\n\nAwng attack with wpsnr = {}".format(wpsnr(img, attacked)))
  elif i==2:
      attacked = blur(img, [1, 1])
      print("\n\nBlur attack with wpsnr = {}".format(wpsnr(img, attacked)))
      """
      plt.figure(figsize=(20, 20))
      plt.subplot(221)
      plt.title('Original')
      plt.imshow(img, cmap='gray')
      plt.subplot(222)
      plt.title('Blur modifications')
      difference = img - attacked
      difference[:8, :8] = 0
      difference[8:16, :8] = 100
      plt.imshow(difference, cmap='gray')

      diff = difference.copy()
      diff[:8, :8] = 0
      diff[8:16, :8] = 100
      attacked = jpeg_compression(img, 35)
      print("\n\nJpeg compression attack with wpsnr = {}".format(wpsnr(img, attacked)))
      plt.subplot(223)
      plt.title('Blur modifications - Jpeg modifications')
      difference = img - attacked
      plt.imshow(diff - difference, cmap='gray')
      plt.subplot(224)
      plt.title('Jpeg modifications')

      difference[:8, :8] = 0
      difference[8:16, :8] = 100
      plt.imshow(difference, cmap='gray')
      plt.show()"""
  elif i==3:
      attacked = sharpening(img, 1, 1)
      print("\n\nSharpening attack with wpsnr = {}".format(wpsnr(img, attacked)))
  elif i==4:
      attacked = median(img, [3, 5])
      print("\n\nMedian filtering attack with wpsnr = {}".format(wpsnr(img, attacked)))
  elif i==5:
      attacked = resizing(img, 6/14)
      print("\n\nResizing attack with wpsnr = {}".format(wpsnr(img, attacked)))
  elif i==6:
      attacked = jpeg_compression(img,11)
      print("\n\nJpeg compression attack with wpsnr = {}".format(wpsnr(img, attacked)))

  if wpsnr(img, attacked) < 35: return random_attack(img)
  return attacked

# FUNCTIONS FOR THE COMPUTATION OF THE ROCK CURVE
def compute_roc(scores, labels):
    # compute ROC
    fpr, tpr, tau = roc_curve(np.asarray(labels), np.asarray(scores), drop_intermediate=False)
    # compute AUC
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2

    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    idx_tpr = np.where((fpr - 0.1) == min(i for i in (fpr - 0.1) if i > 0))
    print('For a FPR approximately equals to 0.1 corresponds a TPR equals to %0.2f' % tpr[idx_tpr[0][0]])
    print('For a FPR approximately equals to 0.1 corresponds a threshold equals to %0.2f' % tau[idx_tpr[0][0]])
    print('Check FPR %0.2f' % fpr[idx_tpr[0][0]])


def create_scores_labels(n_samples=20):
    imageList = []
    for i in range(101):
        imageList.append('sample-images/{num:04d}.bmp'.format(num=i))

    # scores and labels are two lists we will use to append the values of similarity and their labels
    # In scores we will append the similarity between our watermarked image and the attacked one,
    # or  between the attacked watermark and a random watermark
    # In labels we will append the 1 if the scores was computed between the watermarked image and the attacked one,
    # and 0 otherwise
    scores = []
    labels = []

    sample = 0
    while sample < n_samples:
        im = imageList[np.random.randint(101)]
        # Embed Watermark
        mark, watermarked = embedding(im)
        cv2.imwrite("watermarked.bmp", watermarked)
        # fakemark is the watermark for H0
        fakemark = np.random.uniform(0.0, 1.0, mark_size)
        fakemark = np.uint8(np.rint(fakemark))
        # random attack to watermarked image
        res_att = random_attack(watermarked)
        cv2.imwrite("res_att.bmp",res_att)
        # extract attacked watermark
        w_ex = ROCdetection(im, "watermarked.bmp" ,"res_att.bmp")
        # compute similarity H1
        sim = similarity(mark, w_ex)
        print("similarity = {}\n\n".format(sim))
        scores.append(sim)
        labels.append(1)
        # compute similarity H0
        scores.append(similarity(fakemark, w_ex))
        labels.append(0)
        sample += 1
        print(similarity(fakemark, w_ex))
    return scores, labels


scores, labels = create_scores_labels(30)

compute_roc(scores, labels)