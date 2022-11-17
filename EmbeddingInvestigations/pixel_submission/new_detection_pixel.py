
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
    if d > 2.5: return 0 # we can't trust values that are too much far away
    elif d<0.7: return 1 # we can fully trust values that are in [0,1]
    elif d < 1: return 0.7
    elif d < 1.5: return 0.4
    else: return 0.2

def detection(input1, input2, input3):

    #opening images
    original = cv2.imread(input1, 0)
    watermarked = cv2.imread(input2, 0)
    attacked = cv2.imread(input3, 0)

    # choosing parameters
    threshold = 12.59
    blocksize = 8  # it could be a divisor of 480 = 5 * 3 * 2^5, better if a multiple of 8
    n_blocks = 512 // blocksize  # number of blocks
    mark_size = 1024
    mark_repetitions = 1  # 1,2 or 3 are the possible values
    key = 1090  # key for the random shuffle
    random.seed(key)
    position_list = [(blocksize * i, blocksize * j, 2) for i in range(n_blocks) for j in range(n_blocks)]
    random.shuffle(position_list)
    for i in range(140):
        position_list[i] = (position_list[i][0],position_list[i][1],1)


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
    our_mark = np.zeros(mark_size)
    for i in range(mark_size):
        retrieved_bits = []
        confidence = []
        for j in range(mark_repetitions):
            bit = float(redundant_mark[j * mark_size + i])
            retrieved_bits.append(bit)
            confidence.append(compute_confidence(abs(bit - 0.5)))
        if sum(confidence) < 10 ** (-12):
            our_mark[i] = 0
        else:
            retrieved_bits = np.clip(retrieved_bits, 0, 1)  # excluding values more than 1 or less than 0
            our_mark[i] = sum([retrieved_bits[i] * confidence[i] for i in range(len(retrieved_bits))]) / sum(
                confidence)


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
    extracted_mark = np.zeros(mark_size)
    for i in range(mark_size):
        retrieved_bits = []
        confidence = []
        for j in range(mark_repetitions):
            bit = float(redundant_mark[j * mark_size + i])
            retrieved_bits.append(bit)
            confidence.append(compute_confidence(abs(bit - 0.5)))
        if sum(confidence) < 10 ** (-12):
            extracted_mark[i] = 0
        else:
            retrieved_bits = np.clip(retrieved_bits, 0, 1)  # excluding values more than 1 or less than 0
            extracted_mark[i] = sum([retrieved_bits[i] * confidence[i] for i in range(len(retrieved_bits))]) / sum(
                confidence)

    output1 = int(similarity(our_mark,extracted_mark) >= threshold)
    output2 = wpsnr(watermarked, attacked)

    return output1, output2