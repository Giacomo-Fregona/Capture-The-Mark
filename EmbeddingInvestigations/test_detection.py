import os
import time
import importlib
import cv2

from ATTACKS.test_awgn import awgn
from ATTACKS.test_blur import blur
from ATTACKS.test_jpeg import jpeg_compression
from ATTACKS.test_median import median
from ATTACKS.test_resize import resizing
from ATTACKS.test_sharpening import sharpening

imName='lena';
groupname='pixel';
ext = 'bmp';

mod = importlib.import_module('detection_groupName', 'detection_groupName')

original = '%s.%s' % (imName, ext)
watermarked = '%s_%s.%s' % (imName, groupname, ext)

# TIME REQUIREMENT: the detection should run in < 5 seconds
start_time = time.time()
tr, w = mod.detection(original, watermarked, watermarked)
end_time = time.time()
if (end_time - start_time) > 5:
    print('ERROR! Takes too much to run: '+str(end_time - start_time))  

# THE WATERMARK MUST BE FOUND IN THE WATERMARKED IMAGE
if tr == 0:
    print('ERROR! Watermark not found in watermarked image');

# THE WATERMARK MUST NOT BE FOUND IN ORIGINAL
tr, w = mod.detection(original, watermarked, original)
if tr == 1:
    print('ERROR! Watermark found in original');


# CHECK DESTROYED IMAGES
img = cv2.imread(watermarked, 0)
attacked = []
c = 0
ws = []

attacked.append(blur(img, 15))
attacked.append(awgn(img, 50, 123))
attacked.append(resizing(img, 0.1))

for i, a in enumerate(attacked):
    aName = 'attacked-%d.bmp' % i
    cv2.imwrite(aName, a)
    tr, w = mod.detection(original, watermarked, aName)
    if tr == 1:
        c += 1
        ws.append(w)
if c > 0:
    print('ERROR! Watermark found in %d destroyed images with ws %s' % (c, str(ws)))


# CHECK UNRELATED IMAGES
files = [os.path.join('TESTImages', f) for f in os.listdir('TESTImages')]
c = 0
for f in files:
    tr, w = mod.detection(original, watermarked, f)
    if tr == 1:
        c += 1
if c > 0:
    print('ERROR! Watermark found in %d unrelated images' % c)



"""
Please, also check that the mark you extract from the watermarked image is consistent with the one given using this function


def check_mark(X, X_star):
  X_star = np.rint(abs(X_star)).astype(int)
  res = [1 for a, b in zip(X, X_star) if a==b]
  if sum(res) != 1024:
    print('The marks are different, please check your code')
  print(sum(res))
  
check_mark(mark, w_ex)

"""