from random import random

import cv2

import numpy as np
from new_embedding_pixel import invert
from detection_definitive import wpsnr
from cv2 import IMREAD_GRAYSCALE




mark = np.load("pixel.npy")
mark = invert(mark)
lena = cv2.imread('lena.bmp', IMREAD_GRAYSCALE)
blocksize = 8
n_blocks = 64
max = 57.2
index = -1

for l in range(705,10000):
    key = 1090
    np.random.seed(l)
    position_list = [(blocksize * i, blocksize * j, 2) for i in range(n_blocks) for j in range(n_blocks)]
    np.random.shuffle(position_list)
    watermarked = lena.copy()
    for k in range(1024):  # in each step of the loop we insert a bit of the mark
        i, j, alpha = position_list[
            k]  # getting the position of the block and the parameter alpha (which is the number we will add, probably 1 all the times except for the borders)
        if (mark[k % 1024] == 1):  # if the bit of the mark is 0 we do not do anything
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
    if(wpsnr(lena, watermarked)) > max:
        print(l)
        max = wpsnr(lena, watermarked)

