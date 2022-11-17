import os
from math import sqrt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.fft import dct, idct
from sklearn.metrics import roc_curve, auc


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


lena = Image.open('lena.bmp')
lena = np.asarray(Image.open('lena.bmp'), dtype=np.uint8)

blocksize = 8# it could be a divisor of 480 = 5 * 3 * 2^5, better if a multiple of 8
n_blocks = 480//blocksize

mark_size = 1024
mark_repetitions = 2
beta = 10
v = 'multiplicative'

real_mark =  np.random.uniform(0.0, 1.0, mark_size)
real_mark = np.uint8(np.rint(real_mark))


def altEmbedding(image, mark_size, beta, v='multiplicative'):
    mark = np.random.uniform(0.0, 1.0, n_blocks**2)
    global real_mark
    mark[:mark_size]= real_mark.copy()
    mark[mark_size*mark_repetitions:] = 0
    for i in range(mark_size):
        for j in range(1,mark_repetitions):
            mark[j*mark_size+i]=mark[i]
    mark = np.uint8(np.rint(mark))
    #mark = np.zeros_like(mark)
    mark = mark.reshape((n_blocks,n_blocks))
    watermarked = image.copy()
    watermarked += beta*mark
    mark = mark.reshape(n_blocks**2)
    return mark[:mark_size], watermarked

def altDetection(image, watermarked, beta, mark_size, v='multiplicative',mark=np.zeros(mark_size)):
    global real_mark
    extended_mark = ((watermarked-image)/beta).reshape(n_blocks**2)
    """for i in extended_mark:
        if i>1/2:
            i = 1
        else:
            i = 0"""
    for i in range(mark_size):
        for j in range(1,mark_repetitions):
            if (np.random.randint(0,25))<1:
                print("The right mark is {}, our bits are {} and {}".format(real_mark[i],extended_mark[i],extended_mark[i+mark_size]))
            extended_mark[i]+=extended_mark[j*mark_size+i]
    return (extended_mark/mark_repetitions)[:mark_size]


def SSembedding(image, mark_size, beta, v='multiplicative'):
    # Get the DCT transform of the image
    ori_dct = DCT(IDCT(DCT(image)))

    # Get the locations of the most perceptually significant components
    sign = np.sign(ori_dct)#tenerne traccia per poter poi ricostruire l'immagine (boh)
    ori_dct = abs(ori_dct)
    locations = np.argsort(-ori_dct, axis=None) # - sign is used to get descending order
    rows = image.shape[0]
    locations = [(val//rows, val%rows) for val in locations] # locations as (x,y) coordinates

    # Generate a watermark
    mark = np.random.uniform(0.0, 1.0, mark_size)
    mark = np.uint8(np.rint(mark))
    #mark = np.zeros(1024)
    np.save('mark.npy', mark)

    # Embed the watermark
    watermarked_dct = ori_dct.copy()
    for idx, (loc,mark_val) in enumerate(zip(locations, mark)):
        if v == 'additive':
            watermarked_dct[loc] += (beta * mark_val)
        elif v == 'multiplicative':
            watermarked_dct[loc] *= 1 + ( beta * mark_val)

    # Restore sign and o back to spatial domain
    watermarked_dct *= sign
    watermarked = IDCT(watermarked_dct)

    return mark, watermarked


def SSdetection(image, watermarked, beta, mark_size, v='multiplicative'):
    ori_dct = DCT(IDCT(DCT(image)))
    wat_dct = DCT(watermarked)

    # Get the locations of the most perceptually significant components
    ori_dct = abs(ori_dct)
    wat_dct = abs(wat_dct)
    locations = np.argsort(-ori_dct, axis=None)  # - sign is used to get descending order
    rows = image.shape[0]
    locations = [(val // rows, val % rows) for val in locations]  # locations as (x,y) coordinates

    # Preallocate the watermark
    w_ex = np.zeros(mark_size, dtype=np.float64)

    # Extract the watermark
    for idx, loc in enumerate(locations[1:mark_size + 1]):
        if v == 'additive':
            w_ex[idx] = (wat_dct[loc] - ori_dct[loc]) / beta
        elif v == 'multiplicative':
            w_ex[idx] = (wat_dct[loc] - ori_dct[loc]) / (beta * ori_dct[loc])

    return w_ex


def image_to_blocksDC(image):
    blocks_DC_image = np.zeros((n_blocks, n_blocks))
    for i in range(n_blocks):  # cycling on the rows
        for j in range(n_blocks):  # cycling on the columns
            block = image[16 + blocksize * i:16 + blocksize * (i + 1),
                    16 + blocksize * j:16 + blocksize * (j + 1)].copy()  # extracting the block
            blocks_DC_image[i, j] = DCT(IDCT(DCT(block)))[0, 0]  # adding the block value to blocks_DC_image
    return blocks_DC_image


def change_blocksDC(image,wd_blocksDC_image):
    watermarked = image.copy()
    for i in range(n_blocks):
        for j in range(n_blocks):
            block = image[16 + blocksize * i:16 + blocksize * (i + 1),16 + blocksize * j:16 + blocksize * (j + 1)].copy()
            block_dct = DCT(block)
            block_dct[0, 0] = wd_blocksDC_image[i, j]
            block = np.uint8(IDCT(block_dct))
            watermarked[16 + blocksize * i:16 + blocksize * (i + 1),16 + blocksize * j:16 + blocksize * (j + 1)] = block
    return watermarked


def embedding(image, mark_size, beta, v='multiplicative'):

    blocksDC_image = image_to_blocksDC(image)

    mark, wd_blocksDC_image = altEmbedding(blocksDC_image,mark_size,beta,v)#SSembedding(blocksDC_image,mark_size,beta,v)

    watermarked = change_blocksDC(image,wd_blocksDC_image)

    return mark, watermarked


def detection(image, watermarked, beta, mark_size, v='multiplicative'):
    blocksDC_image = image_to_blocksDC(image)
    blocksDC_watermarked = image_to_blocksDC(watermarked)
    w_ex = altDetection(blocksDC_image,blocksDC_watermarked, beta, mark_size,v)#SSdetection(blocksDC_image,blocksDC_watermarked, beta, mark_size,v)
    return w_ex


mark, wd = embedding(lena, mark_size,beta,v)

"""show(lena)
show(wd)"""

print('wPSNR: %.2fdB' % wpsnr(lena, wd))

w_ex = detection(lena, wd, beta, mark_size,v)

def similarity(X, X_star):
  # Computes the similarity measure between the original and the new watermarks.
  s = np.sum(np.multiply(X, X_star)) / np.sqrt(np.sum(np.multiply(X_star, X_star)))
  return s

print(similarity(mark, w_ex))





#this seed was set just to make you obtain the same result
np.random.seed(3)
def awgn(img, std, seed):
  mean = 0.0   # some constant
  #np.random.seed(seed)
  attacked = img + np.random.normal(mean, std, img.shape)
  attacked = np.clip(attacked, 0, 255)
  return attacked

def blur(img, sigma):
  from scipy.ndimage.filters import gaussian_filter
  attacked = gaussian_filter(img, sigma)
  return attacked

def sharpening(img, sigma, alpha):
  import scipy
  from scipy.ndimage import gaussian_filter
  import matplotlib.pyplot as plt

  #print(img/255)
  filter_blurred_f = gaussian_filter(img, sigma)

  attacked = img + alpha * (img - filter_blurred_f)
  return attacked

def median(img, kernel_size):
  from scipy.signal import medfilt
  attacked = medfilt(img, kernel_size)
  return attacked

def resizing(img, scale):
  from skimage.transform import rescale
  x, y = img.shape
  attacked = rescale(img, scale)
  attacked = rescale(attacked, 1/scale)
  attacked = attacked[:x, :y]
  return attacked

def jpeg_compression(img, QF):
  from PIL import Image
  img = Image.fromarray(img)
  img.save('tmp.jpg',"JPEG", quality=QF)
  attacked = Image.open('tmp.jpg')
  attacked = np.asarray(attacked,dtype=np.uint8)
  os.remove('tmp.jpg')

  return attacked

def random_attack(img):
  i = np.random.randint(1,7)
  if i==1:
      print("\n\nawng attack")
      attacked = awgn(img, 5.0, 123)
  elif i==2:
      print("\n\nblur attack")
      attacked = blur(img, [3, 2])
  elif i==3:
      print("\n\nsharpening attack")
      attacked = sharpening(img, 1, 1)
  elif i==4:
      print("\n\nmedian attack")
      attacked = median(img, [3, 5])
  elif i==5:
      print("\n\nresizing attack")
      attacked = resizing(img, 0.5)
  elif i==6:
      print("\n\njpeg attack")
      attacked = jpeg_compression(img, 75)
  show(attacked)
  return attacked

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

# Read image
# img = cv2.imread('images/lena_pixel.bmp', 0)
# print('Image shape: ', im.shape)

imageList = []
for i in range(101):
  imageList.append(np.asarray(Image.open('sample-images/{num:04d}.bmp'.format(num = i)), dtype=np.uint8))

#scores and labels are two lists we will use to append the values of similarity and their labels
#In scores we will append the similarity between our watermarked image and the attacked one,
# or  between the attacked watermark and a random watermark
#In labels we will append the 1 if the scores was computed between the watermarked image and the attacked one,
#and 0 otherwise
scores = []
labels = []



sample = 0
while sample<20:
    im = imageList[np.random.randint(101)]
    #Embed Watermark
    mark, watermarked = embedding(im, mark_size, beta, v)
    #fakemark is the watermark for H0
    fakemark = np.random.uniform(0.0, 1.0, mark_size)
    fakemark = np.uint8(np.rint(fakemark))
    #random attack to watermarked image
    res_att = random_attack(watermarked)
    #extract attacked watermark
    w_ex = detection(im, res_att, beta, mark_size, v)
    # def detection(image, watermarked, alpha, mark_size, v='multiplicative'):
    #compute similarity H1
    scores.append(similarity(mark, w_ex))
    labels.append(1)
    #compute similarity H0
    scores.append(similarity(fakemark, w_ex))
    labels.append(0)
    sample += 1

#print(scores)
compute_roc(scores, labels)