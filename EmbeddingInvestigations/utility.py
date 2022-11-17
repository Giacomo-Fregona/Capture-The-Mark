from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.fft import dct, idct
import os.path as path

def get_probabilities(blocksize=8):
    print("Calculating probabilities...")
    if path.exists("prob_ones.npy"):
        prob_ones = np.load('prob_ones.npy')
        print(prob_ones)
    else:
        n_blocks = 512//blocksize

        image = np.zeros((512, 512))
        image_copy = image.copy()
        matrix = np.zeros((n_blocks, n_blocks))

        for i in range(n_blocks//2):  # cycling on half of the rows (to increase performance)
            for j in range(n_blocks//2):  # cycling on half the columns  (to increase performance)
                image_copy[blocksize * i:blocksize * (i + 1), blocksize * j:blocksize * (j + 1)] = np.ones((8, 8))
                if j > 1 and matrix[i, j-1] == matrix[i, j-2]:
                    matrix[i, j] = matrix[i, j-1]
                elif i > 1 and matrix[i-1, j] == matrix[i-2, j]:
                    matrix[i, j] = matrix[i-1, j]
                else:
                    matrix[i, j] = round(wpsnr(image, image_copy), 2)
                image_copy = image.copy()

        # Merging the symmetric matrixes to have a complete one 
        matrix_reverse = np.flip(matrix, axis=0)
        matrix_reverse2 = np.flip(matrix, axis=1)
        matrix_reverse3 = np.flip(matrix_reverse, axis=1)
        matrix = matrix + matrix_reverse + matrix_reverse2 + matrix_reverse3 # Complete matrix

        # Probability calculation: gives a percentage for each cell of the matrix based on its wPSNR and a max value of 30% more than the maximum of the matrix (to avoid corners being at 100%)
        prob_ones = (matrix/np.amax(matrix))
        np.save('prob_ones.npy', prob_ones)
    return prob_ones

def show(image):
    plt.imshow(image,cmap='gray')
    plt.show()

def DCT(image):
    return dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')

def IDCT(image):
    return idct(idct(image, axis=1, norm='ortho'), axis=0, norm='ortho')

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

def similarity(X, X_star):
  # Computes the similarity measure between the original and the new watermarks.
  s = np.sum(np.multiply(X, X_star)) / np.sqrt(np.sum(np.multiply(X_star, X_star)))
  return s