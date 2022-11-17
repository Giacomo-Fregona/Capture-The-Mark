
import random

import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc


def embedding(image_name):

    image = cv2.imread(image_name, 0)

    blocksize = 8  # it could be a divisor of 480 = 5 * 3 * 2^5, better if a multiple of 8
    n_blocks = 512 // blocksize
    mark_size = 1024
    mark_repetitions = 1
    key = 4187
    random.seed(key) # key for the random shuffle
    position_list = [(blocksize * i, blocksize * j, 1) for i in range(n_blocks) for j in range(n_blocks)]
    random.shuffle(position_list)

    # retrieving the watermark
    mark = np.load("pixel.npy")

    # inserting the watermark
    watermarked = image.copy()
    for k in range(mark_repetitions * mark_size):  # in each step of the loop we insert a bit of the mark
        i, j, alpha = position_list[
            k]  # getting the position of the block and the parameter alpha (which is the number we will add, probably 1 all the times except for the borders)
        if (mark[k % mark_size] == 1):  # if the bit of the mark is 0 we do not do anything
            block = watermarked[i:i + blocksize, j:j + blocksize]

            # Here there was the problem: if we have 255 in the pixel, moving up by just one would have created problems. (here we talk about +1 but more properly we should talk about +alpha )
            # So first of all we compute the mean value of the block and decide if we are closer to 0 or to 255. Then we add alpha if we are closer to 0 and we subtract alpha if we are closer to 255

            if block.mean() < 128:  # In that case we hope there will not be pixels with values near to 255
                for ii in range(blocksize):
                    for jj in range(blocksize):
                        block[ii, jj] += min(alpha, 255 - block[
                            ii, jj])  # this min takes care of the fact that there could be both 0 and 255 in the block truncating the added value
            else:  # In that case we hope there will not be pixels with values near to 0
                for ii in range(blocksize):
                    for jj in range(blocksize):
                        block[ii, jj] -= min(alpha, block[
                            ii, jj])  # this min takes care of the fact that there could be both 0 and 255 in the block truncating the subtracted value
    return mark, watermarked
