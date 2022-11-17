import re
from turtle import position
from cv2 import IMREAD_GRAYSCALE
import numpy as np
#from PIL import Image
from sklearn.metrics import roc_curve, auc
import cv2
from attacks import *
from utility import *


# CHOOSING THE PARAMETERS
blocksize = 8 # it could be a divisor of 480 = 5 * 3 * 2^5, better if a multiple of 8
n_blocks = 512//blocksize
mark_size = 1024
mark_repetitions = 2

#CHOOSING THE MAR
real_mark = np.load("pixel.npy")
print("Weight of the mark = {}".format(real_mark.sum()))


#list of the positions and security values in which to insert (and retrieve) the watermark bits. Security value = number that we add to each bit in case we are representing a one
probabilities = get_probabilities(blocksize)
position_list = [(blocksize*i,blocksize*j, probabilities[i,j]) for i in range(n_blocks) for j in range(n_blocks)]


#EMBEDDING
def embedding(image):

    #retrieving the watermark from the global variable realmark... to substitute with loading our mark
    global real_mark
    mark = real_mark.copy()

    #inserting the watermark
    watermarked = image.copy()
    for k in range(mark_repetitions*mark_size):#in each step of the loop we insert a bit of the mark
        i,j,alpha = position_list[k]# getting the position of the block and the parameter alpha (which is the number we will add, probably 1 all the times except for the borders)
        if (mark[k%mark_size]==1):#if the bit of the mark is 0 we do not do anything
            block = watermarked[i:i+blocksize,j:j+blocksize]

            #Here there was the problem: if we have 255 in the pixel, moving up by just one would have created problems. (here we talk about +1 but more properly we should talk about +alpha )
            #So first of all we compute the mean value of the block and decide if we are closer to 0 or to 255. Then we add alpha if we are closer to 0 and we subtract alpha if we are closer to 255

            if block.mean()<128:# In that case we hope there will not be pixels with values near to 255
                for ii in range(blocksize):
                    for jj in range(blocksize):
                        block[ii,jj]+= min(alpha, 255-block[ii,jj])#this min takes care of the fact that there could be both 0 and 255 in the block truncating the added value
            else: # In that case we hope there will not be pixels with values near to 0
                for ii in range(blocksize):
                    for jj in range(blocksize):
                        block[ii,jj]-= min(alpha, block[ii,jj])#this min takes care of the fact that there could be both 0 and 255 in the block truncating the subtracted value
    return mark, watermarked


#DETECTION
def detection(image, watermarked):

    redundant_mark = np.zeros(mark_repetitions*mark_size)#bits of the watermark multiple copies extracted

    for k in range(mark_repetitions * mark_size):
        i, j, alpha = position_list[k]
        block_watermarked = watermarked[i:i + blocksize, j:j + blocksize].copy()
        block_image = image[i:i + blocksize, j:j + blocksize].copy()
        block_watermarked = block_watermarked.astype('float64')# going to float excludes overflow problems in the lines below
        block_image = block_image.astype('float64')

        #How to design a strategy with the new embedding? We sum all the pixel differences between watermarked and original image (score) and
        #divide by the nuber of values we should have added/subtracted if there was a 1 on the mark (divisor)

        divisor = 0
        if block_image.mean() < 128:  # In that case we should look for added alpha
            score = (block_watermarked-block_image).sum()
            for ii in range(blocksize):
                for jj in range(blocksize):
                    divisor += min(alpha, 255 - block_image[ii, jj])
        else:  # In that case we should look for subtracted alpha
            score = ( block_image - block_watermarked).sum()
            for ii in range(blocksize):
                for jj in range(blocksize):
                    divisor += min(alpha, block_image[ii, jj])
        redundant_mark[k] = score / divisor


    redundant_mark = np.clip(redundant_mark, 0, 1) #excluding values more than 1 or less than 0

    #calculating and returning the mean value of the bits that we got from different copies
    for i in range(mark_size):
        for j in range(1, mark_repetitions):
            redundant_mark[i] += redundant_mark[j * mark_size + i]
    return (redundant_mark / mark_repetitions)[:mark_size]



#AN EXAMPLE OF EMBEDDING

#lena = Image.open('lena_pixel.bmp')
#lena = np.asarray(Image.open('lena_pixel.bmp'), dtype=np.uint8)

lena = cv2.imread('lena_pixel.bmp', IMREAD_GRAYSCALE)
#print(type(lena))
#print(lena.shape)
#print(lena)


mark, wd = embedding(lena)

print('wPSNR for lena = %.2fdB' % wpsnr(lena, wd))

w_ex = detection(lena, wd)

print("similarity for lena = {}".format(similarity(mark, w_ex)))



#FUNCTIONS FOR THE COMPUTATION OF THE ROCK CURVE
def compute_roc(scores, labels):
    #compute ROC
    fpr, tpr, tau = roc_curve(np.asarray(labels), np.asarray(scores), drop_intermediate=False)
    #compute AUC
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2

    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    idx_tpr = np.where((fpr-0.05)==min(i for i in (fpr-0.05) if i > 0))
    print('For a FPR approximately equals to 0.05 corresponds a TPR equals to %0.2f' % tpr[idx_tpr[0][0]])
    print('For a FPR approximately equals to 0.05 corresponds a threshold equals to %0.2f' % tau[idx_tpr[0][0]])
    print('Check FPR %0.2f' % fpr[idx_tpr[0][0]])

def create_scores_labels(n_samples = 50):
    mean_wpsnr=[]
    imageList = []
    for i in range(101):
        imageList.append(cv2.imread('sample-images/{num:04d}.bmp'.format(num = i), IMREAD_GRAYSCALE))

      
      

    #scores and labels are two lists we will use to append the values of similarity and their labels
    #In scores we will append the similarity between our watermarked image and the attacked one,
    # or  between the attacked watermark and a random watermark
    #In labels we will append the 1 if the scores was computed between the watermarked image and the attacked one,
    #and 0 otherwise
    scores = []
    labels = []

    sample = 0
    while sample<n_samples:
        im = imageList[np.random.randint(101)]
        #Embed Watermark
        mark, watermarked = embedding(im)
        mean_wpsnr.append(wpsnr(im,watermarked))
        #fakemark is the watermark for H0
        fakemark = np.random.uniform(0.0, 1.0, mark_size)
        fakemark = np.uint8(np.rint(fakemark))
        #random attack to watermarked image
        res_att = random_attack(watermarked)
        #extract attacked watermark
        w_ex = detection(im, res_att)
        # def detection(image, watermarked, alpha, mark_size, v='multiplicative'):
        #compute similarity H1
        scores.append(similarity(mark, w_ex))
        labels.append(1)
        #compute similarity H0
        scores.append(similarity(fakemark, w_ex))
        labels.append(0)
        sample += 1
    print("\n\n\nTRESHOLD COMPUTATION\n\naverage wPSNR: ", np.mean(mean_wpsnr))
    return scores, labels


scores, labels = create_scores_labels(10)

compute_roc(scores, labels)