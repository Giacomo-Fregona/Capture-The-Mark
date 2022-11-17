import cv2
from wpsnr import wpsnr
import random
import numpy as np

def similarity(X, X_star):
  # Computes the similarity measure between the original and the new watermarks.
  norm = np.sum(np.multiply(X_star, X_star))
  if norm < 10**(-12):
      return 0
  s = np.sum(np.multiply(X, X_star)) / np.sqrt(norm)
  return s


def detection(input1, input2, input3):

    #opening images
    original = cv2.imread(input1, 0)
    watermarked = cv2.imread(input2, 0)
    attacked = cv2.imread(input3, 0)

    #choosing parameters
    threshold = 14.1
    blocksize = 8  # it could be a divisor of 480 = 5 * 3 * 2^5, better if a multiple of 8
    n_blocks = 512 // blocksize
    mark_size = 1024
    mark_repetitions = 2
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

    output1 = similarity(our_mark,extracted_mark) >= threshold
    output2 = wpsnr(watermarked, attacked)
    return output1, output2