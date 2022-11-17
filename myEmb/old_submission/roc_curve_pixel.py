import os
from utility import wpsnr
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt
from cv2 import resize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import cv2
from wpsnr import wpsnr
import random
import numpy as np
from embedding_pixel import embedding

def similarity(X, X_star):
  # Computes the similarity measure between the original and the new watermarks.
  norm = np.sum(np.multiply(X_star, X_star))
  if norm < 10**(-12):
      return 0
  s = np.sum(np.multiply(X, X_star)) / np.sqrt(norm)
  return s

def ROCdetection(input1, input2, input3):# detection of the code with only difference that the similarity is returned
    #opening images
    original = cv2.imread(input1, 0)
    watermarked = cv2.imread(input2, 0)
    attacked = cv2.imread(input3, 0)

    #choosing parameters
    threshold = 14.1
    blocksize = 8  # it could be a divisor of 480 = 5 * 3 * 2^5, better if a multiple of 8
    n_blocks = 512 // blocksize
    mark_size = 1024
    mark_repetitions = 1
    key = 4187
    random.seed(key)
    position_list = [(blocksize * i, blocksize * j, 1) for i in range(n_blocks) for j in range(n_blocks)]
    random.shuffle(position_list)

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

    redundant_mark = np.clip(redundant_mark, 0, 1)  # excluding values more than 1 or less than 0

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

    redundant_mark = np.clip(redundant_mark, 0, 1)  # excluding values more than 1 or less than 0

    # calculating and returning the mean value of the bits that we got from different copies
    for i in range(mark_size):
        for j in range(1, mark_repetitions):
            redundant_mark[i] += redundant_mark[j * mark_size + i]
    extracted_mark = (redundant_mark / mark_repetitions)[:mark_size]

    return extracted_mark


blocksize = 8  # it could be a divisor of 480 = 5 * 3 * 2^5, better if a multiple of 8
n_blocks = 512 // blocksize
mark_size = 1024
mark_repetitions = 2

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


def create_scores_labels(n_samples=50):
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
        scores.append(sim)
        labels.append(1)
        # compute similarity H0
        scores.append(similarity(fakemark, w_ex))
        labels.append(0)
        sample += 1
    return scores, labels

scores, labels = create_scores_labels(20)

compute_roc(scores, labels)