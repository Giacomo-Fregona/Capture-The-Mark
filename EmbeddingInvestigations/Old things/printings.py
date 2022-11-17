import os
from math import sqrt

import numpy as np
import scipy.linalg
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import cv2
from scipy.signal import convolve2d
from skimage import restoration
from scipy.fft import dct, idct

csf = np.genfromtxt('csf.csv', delimiter=',')

def show(image):
    plt.imshow(image,cmap='gray')
    plt.show()

def wpsnr(img1, img2):
  img1 = np.float32(img1)/255.0
  img2 = np.float32(img2)/255.0

  difference = img1-img2
  same = not np.any(difference)
  if same is True:
      return 9999999
  csf = np.genfromtxt('csf.csv', delimiter=',')
  ew = convolve2d(difference, np.rot90(csf,2), mode='valid')
  decibels = 20.0*np.log10(1.0/sqrt(np.mean(np.mean(ew**2))))
  return decibels

lena = Image.open('lena.bmp')
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

plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.title('wPSNR = %.2f' % wpsnr(lena,lenac1),fontsize = 30)
plt.imshow(lenac1, cmap='gray')
plt.subplot(122)
plt.title('wPSNR = %.2f' % wpsnr(lena,lenac2),fontsize = 30)
plt.imshow(lenac2, cmap='gray')
plt.show()
#show(lena)

mark_size = 1024
alpha = 10
v= 'additive'
blocksize = 8# it could be a divisor of 480 = 5 * 3 * 2^5, better if a multiple of 8
n_blocks = 480//blocksize

"""
dctim = np.zeros((512,512))
for i in range(64):  # cycling on the rows
    for j in range(64):  # cycling on the columns
        block = lena[blocksize * i:blocksize * (i + 1), blocksize * j: blocksize * (j + 1)].copy()  # extracting the block
        show(block)
        plt.imshow(dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho'))
        plt.show()
        dctim[blocksize * i:blocksize * (i + 1), blocksize * j: blocksize * (j + 1)] = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')  # adding the block value to blocks_DC_image
"""
def image_to_blockDC(image):
    blocks_DC_image = np.zeros((n_blocks, n_blocks))
    for i in range(n_blocks):  # cycling on the rows
        for j in range(n_blocks):  # cycling on the columns
            block = image[16 + blocksize * i:16 + blocksize * (i + 1),
                    16 + blocksize * j:16 + blocksize * (j + 1)].copy()  # extracting the block
            blocks_DC_image[i, j] = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')[0, 0]  # adding the block value to blocks_DC_image
    return blocks_DC_image

dctim = image_to_blockDC(lena)

plt.imshow(dctim)
plt.show()

divided =255 * np.ones((550, 550))
divided[20:532,20:532] = lena.copy()
for i in range(65):
    divided[:,20 + i*blocksize] = 0

for i in range(65):
    divided[20 + i * blocksize, : ] = 0


show(divided)






def SSembedding(image, mark_size, alpha, v='multiplicative'):
    # Get the DCT transform of the image
    ori_dct = dct(dct(image,axis=0, norm='ortho'),axis=1, norm='ortho')

    # Get the locations of the most perceptually significant components
    sign = np.sign(ori_dct)#tenerne traccia per poter poi ricostruire l'immagine (boh)
    ori_dct = abs(ori_dct)
    locations = np.argsort(-ori_dct,axis=None) # - sign is used to get descending order
    rows = image.shape[0]
    locations = [(val//rows, val%rows) for val in locations] # locations as (x,y) coordinates

    # Generate a watermark
    mark = np.random.uniform(0.0, 1.0, mark_size)
    mark = np.uint8(np.rint(mark))
    np.save('mark.npy', mark)

    # Embed the watermark
    watermarked_dct = ori_dct.copy()
    for idx, (loc,mark_val) in enumerate(zip(locations, mark)):
        if v == 'additive':
            watermarked_dct[loc] += (alpha * mark_val)
        elif v == 'multiplicative':
            watermarked_dct[loc] *= 1 + ( alpha * mark_val)

    # Restore sign and o back to spatial domain
    watermarked_dct *= sign
    watermarked = np.uint8(idct(idct(watermarked_dct,axis=1, norm='ortho'),axis=0, norm='ortho'))

    return mark, watermarked




def image_to_blockDC(image):
    blocks_DC_image = np.zeros((n_blocks, n_blocks))
    for i in range(n_blocks):  # cycling on the rows
        for j in range(n_blocks):  # cycling on the columns
            block = image[16 + blocksize * i:16 + blocksize * (i + 1),
                    16 + blocksize * j:16 + blocksize * (j + 1)].copy()  # extracting the block
            blocks_DC_image[i, j] = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')[0, 0]  # adding the block value to blocks_DC_image
    return blocks_DC_image


def embedding(image, mark_size, alpha, v='multiplicative'):

    blocks_DC_image = image_to_blockDC(image)


    mark, wd_blocks_DC_image = SSembedding(blocks_DC_image,mark_size,alpha,v)

    # Getting the DCT transform of DCimage
    ori_dct = dct(dct(blocks_DC_image, axis=0, norm='ortho'), axis=1, norm='ortho')


    # Get the locations of the most perceptually significant components
    sign = np.sign(ori_dct)  # tenerne traccia per poter poi ricostruire l'immagine (boh)
    ori_dct = abs(ori_dct)
    locations = np.argsort(-ori_dct, axis=None)  # - sign is used to get descending order
    rows = blocks_DC_image.shape[0]
    locations = [(val // rows, val % rows) for val in locations]  # locations as (x,y) coordinates

    # Generate a watermark
    mark = np.random.uniform(0.0, 1.0, mark_size)
    mark = np.uint8(np.rint(mark))
    #mark = np.zeros((mark_size))
    np.save('mark.npy', mark)



    # Embed the watermark
    watermarked_dct_blocks_DC_image = ori_dct.copy()
    for idx, (loc, mark_val) in enumerate(zip(locations, mark)):
      if v == 'additive':
        watermarked_dct_blocks_DC_image[loc] += (alpha * mark_val)
      elif v == 'multiplicative':
        watermarked_dct_blocks_DC_image[loc] *= 1 + (alpha * mark_val)

    # Restore sign and go back to spatial domain
    watermarked_dct_blocks_DC_image *= sign
    watermarked_blocks_DC_image = idct(idct(watermarked_dct_blocks_DC_image, axis=1, norm='ortho'), axis=0, norm='ortho')



    # create a copy of the original image that will be returned as the watermarked image
    watermarked = image.copy()
    for i in range(n_blocks):
      for j in range(n_blocks):
        block = image[16 + blocksize * i:16 + blocksize * (i + 1), 16 + blocksize * j:16 + blocksize * (j + 1)].copy()
        block_dct = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
        block_dct[0,0] = watermarked_blocks_DC_image[i,j]
        block = np.uint8(idct(idct(block_dct, axis=1, norm='ortho'), axis=0, norm='ortho'))
        watermarked[16 + blocksize * i:16 + blocksize * (i + 1), 16 + blocksize * j:16 + blocksize * (j + 1)] = block
    return mark, watermarked




def detection(image, watermarked, alpha, mark_size, v='multiplicative'):

  im = image.copy()
  # nb we discard the first 16 pixels on each border

  # create the DC image
  DCimage = np.zeros((n_blocks, n_blocks))

  for i in range(n_blocks):
    for j in range(n_blocks):
      block = im[16 + blocksize * i:16 + blocksize * (i + 1), 16 + blocksize * j:16 + blocksize * (j + 1)]
      DCimage[i, j] = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')[0, 0]

  imwd = watermarked.copy()
  # nb we discard the first 16 pixels on each border

  # create the DC image
  DCimage = np.zeros((n_blocks, n_blocks))

  for i in range(n_blocks):
    for j in range(n_blocks):
      block = imwd[16 + blocksize * i:16 + blocksize * (i + 1), 16 + blocksize * j:16 + blocksize * (j + 1)]
      DCimage[i, j] = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')[0, 0]




  ori_dct = dct(dct(im, axis=0, norm='ortho'), axis=1, norm='ortho')
  wat_dct = dct(dct(imwd, axis=0, norm='ortho'), axis=1, norm='ortho')

  # Get the locations of the most perceptually significant components
  ori_dct = abs(ori_dct)
  wat_dct = abs(wat_dct)
  locations = np.argsort(-ori_dct, axis=None)  # - sign is used to get descending order
  rows = image.shape[0]
  locations = [(val // rows, val % rows) for val in locations]  # locations as (x,y) coordinates

  # Construct the extracted watermark
  w_ex = np.zeros(mark_size, dtype=np.float64)

  # Extract the watermark
  for idx, loc in enumerate(locations[1:mark_size + 1]):
    if v == 'additive':
      w_ex[idx] = (wat_dct[loc] - ori_dct[loc]) / alpha
    elif v == 'multiplicative':
      w_ex[idx] = (wat_dct[loc] - ori_dct[loc]) / (alpha * ori_dct[loc])

    """for i in range(w_ex.shape[0]):
        print(w_ex[i], w_ex[i]>1/2)
        if (w_ex[i] > 1/2):
            w_ex[i]= 1
        else:
            w_ex[i] = 0

    w_ex = np.uint8(w_ex)"""
    return w_ex




mark, wd = embedding(lena, mark_size,alpha,v)

show(lena)
show(wd)

print('wPSNR: %.2fdB' % wpsnr(lena, wd))

w_ex = detection(lena, wd, alpha, mark_size,v)
print(mark)
print(w_ex)

def similarity(X, X_star):
  # Computes the similarity measure between the original and the new watermarks.
  s = np.sum(np.multiply(X, X_star)) / np.sqrt(np.sum(np.multiply(X_star, X_star)))
  return s

print(similarity(mark, w_ex))

"""plt.imshow(wd,cmap='gray')
plt.show()"""






"""cosa migliorare:
fare il round di w_ex
inserire coding theory
cambiare le frequenze del secondo livello di inserimento
pensare trasformate diverse per il secondo livello (fourier?)
testare attacchi
"""

