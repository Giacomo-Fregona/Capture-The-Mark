import cv2
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
import numpy as np
from math import sqrt
from cv2 import IMREAD_GRAYSCALE

from pixel_submission.embedding_pixel import embedding


def wpsnr(img1, img2):
    img1 = np.float32(img1)/255.0
    img2 = np.float32(img2)/255.0

    difference = img1-img2
    same = not np.any(difference)
    if same is True:
      return 9999999
    csf = np.genfromtxt('csf.csv', delimiter=',')
    ew = convolve2d(difference, np.rot90(csf,2), mode='valid')
    print(ew.shape)

    decibels = 20.0*np.log10(1.0/sqrt(np.mean(np.mean(ew**2))))

    return decibels



from new_wpsnr import new_wpsnr

lena = cv2.imread('lena.bmp', IMREAD_GRAYSCALE)

mark, wd = embedding('lena.bmp')

print('wPSNR for lena = %.2fdB' % new_wpsnr(lena, wd))

wd = lena.copy()
wd[50,50] += 1;
print('wPSNR for lena = %.2fdB' % wpsnr(lena, wd))
print('new_wPSNR for lena = %.2fdB' % wpsnr(lena, wd))

lena = np.asarray(lena)


lenac1 = lena.copy()
lenac2 = lena.copy()

lenac1[:10,:] = 0
lenac1[:,:10] = 0
lenac1[502:,:] = 0
lenac1[:,502:] = 0

lenac2[100:110,:] = 0
lenac2[:,100:110] = 0
lenac2[502-50:512-50,:] = 0
lenac2[:,502-50:512-50] = 0

plt.figure(figsize=(15, 15))
plt.subplot(221)
plt.title('wPSNR = %.2f' % wpsnr(lena,lenac1),fontsize = 20)
plt.imshow(lenac1, cmap='gray')
plt.subplot(222)
plt.title('wPSNR = %.2f' % wpsnr(lena,lenac2),fontsize = 20)
plt.imshow(lenac2, cmap='gray')
plt.subplot(223)
plt.title('new_wPSNR = %.2f' % new_wpsnr(lena,lenac1),fontsize = 20)
plt.imshow(lenac1, cmap='gray')
plt.subplot(224)
plt.title('new_wPSNR = %.2f' % new_wpsnr(lena,lenac2),fontsize = 20)
plt.imshow(lenac2, cmap='gray')
plt.show()

