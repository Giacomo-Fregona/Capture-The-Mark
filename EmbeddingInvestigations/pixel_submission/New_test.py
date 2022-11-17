import cv2
import numpy as np

from new_detection_pixel import detection, wpsnr, similarity
from  new_embedding_pixel import embedding

#AN EXAMPLE OF EMBEDDING

#lena = Image.open('lena_pixel.bmp')
#lena = np.asarray(Image.open('lena_pixel.bmp'), dtype=np.uint8)

lena = cv2.imread('lena.bmp', 0)
#print(type(lena))
#print(lena.shape)
#print(lena)

mark = np.load('pixel.npy')

counter = 0
for i in range(len(mark)):
    if counter == 416:
        print(i)
        break
    counter += mark[i]

"""im = "lena.bmp"
mark, watermarked = embedding(im)
cv2.imwrite("watermarked.bmp",watermarked)
#watermarked[ 100: 121, 77: 88] = 100
#cv2.imwrite("res_att.bmp",watermarked)
print(detection(im, "watermarked.bmp" ,"watermarked.bmp"))"""