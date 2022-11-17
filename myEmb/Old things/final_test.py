from embedding_definitive import *
from detection_definitive import *



im = "lena_pixel.bmp"
mark, watermarked = embedding(im)
cv2.imwrite("watermarked.bmp",watermarked)
watermarked[ 100: 121, 77: 88] = 100
cv2.imwrite("res_att.bmp",watermarked)
print(detection(im, "watermarked.bmp" ,"res_att.bmp"))