import random

import numpy as np

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
    f.write("[{},{}],".format(ones_in_middle[i], zeros_on_border[i]))
    position_list = swap(position_list, ones_in_middle[i], zeros_on_border[i])
    counter += 1
# print("number of swaps = {}".format(counter))

f.close()


