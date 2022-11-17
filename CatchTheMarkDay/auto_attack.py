import numpy as np
import pandas as pd
from all_useful_things import *
import importlib

us = 'pixel'
group_names = ['ef26420c', 'you_shall_not_mark', 'blitz', 'omega', 'howimetyourmark', 'weusedlsb', 'thebavarians', 'theyarethesamepicture', 'dinkleberg', 'failedfouriertransform']

working_on = group_names[0] #change this if you want to work on other groups
#working_on = us# only to test the script
print('Working on {}\n'.format(working_on))

#DETECTION FUNCTION
mod = importlib.import_module('detection_{}'.format(working_on))
detection = mod.detection

#IMAGES Here you need to add manually the names pf the images
image_names = ['buildings', 'tree', 'rollercoaster']
print('Selected images are: {}\n'.format(image_names))

#original images
o0 = cv2.imread('{}.bmp'.format(image_names[0]), 0)
o1 = cv2.imread('{}.bmp'.format(image_names[1]), 0)
o2 = cv2.imread('{}.bmp'.format(image_names[2]), 0)

#watermarked images
images = [0,0,0]
images[0] = cv2.imread('{}_{}.bmp'.format(working_on, image_names[0]), 0)
images[1] = cv2.imread('{}_{}.bmp'.format(working_on, image_names[1]), 0)
images[2] = cv2.imread('{}_{}.bmp'.format(working_on, image_names[2]), 0)


attack_type = ['awgn','blur','sharpening','median','resizing','jpeg_compression']
strings_list = ['low_','mid_','high_']
index_list = []
for i in attack_type:
    if i == 'resizing':
        index_list.append(i)
    else:
        for j in strings_list:
            index_list.append(j+i)

scores = []

df_scores = pd.DataFrame(index=index_list,columns=image_names)

for image in images:

    params_awgn = [1.0,5.0,10.0]
    for i in params_awgn:
        attacked = awgn(image, i, 123)
        scores.append(wpsnr(image, attacked))
    
    params_blur = [[2,1],[3,2],[5,3]]
    for i in params_blur:
        attacked = blur(image, i)
        scores.append(wpsnr(image, attacked))

    params_sharp = [2,1,0.5]
    for i in params_sharp:
        attacked = sharpening(image, i,1)
        scores.append(wpsnr(image, attacked))

    params_median = [[1,3],[3,5],[5,9]]
    for i in params_median:
        attacked = median(image, i)
        scores.append(wpsnr(image, attacked))

    attacked = resizing(image, 4)
    scores.append(wpsnr(image, attacked))
    
    params_jpeg = [75,50,25]
    for i in params_jpeg:
        attacked = jpeg_compression(image, i)
        scores.append(wpsnr(image, attacked))

    df_temp = pd.DataFrame(scores,index=index_list)
    df_scores = df_scores.join(df_temp)

print(df_scores)
