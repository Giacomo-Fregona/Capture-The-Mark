
import random

import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc

from wpsnr import wpsnr


def embedding(image_name):

    image = cv2.imread(image_name, 0)

    blocksize = 8  # it could be a divisor of 480 = 5 * 3 * 2^5, better if a multiple of 8
    n_blocks = 512 // blocksize
    mark_size = 1024
    mark_repetitions = 1
    key = 1090
    random.seed(key)
    position_list = [(blocksize * i, blocksize * j, 2) for i in range(n_blocks) for j in range(n_blocks)]
    random.shuffle(position_list)

    for i in range(200):
        position_list[i] = (position_list[i][0],position_list[i][1],1)

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
    for i in range(1024):
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
    f = open("swaps.txt", "a")

    for i in range(48):
        f.write("[{},{}],".format(ones_in_middle[i],zeros_on_border[i]))
        position_list = swap(position_list, ones_in_middle[i], zeros_on_border[i])
        counter +=1
    #print("number of swaps = {}".format(counter))

    f.close()


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
    print("embedding with wPSNR = {}".format(wpsnr(image,watermarked)))
    return mark, watermarked
