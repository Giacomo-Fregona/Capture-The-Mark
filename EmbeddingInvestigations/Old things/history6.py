import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve, auc


from attacks import *
from utility import *


# CHOOSING THE PARAMETERS
blocksize = 8# it could be a divisor of 480 = 5 * 3 * 2^5, better if a multiple of 8
n_blocks = 480//blocksize
mark_size = 1024
mark_repetitions = 2
beta = 8.49 #scaling factor in the embedding/detection
v = 'additive' # for now the only possible choice


#CREATING A TEST MARK
real_mark =  np.random.uniform(0.0, 1.0, mark_size)
real_mark = np.uint8(np.rint(real_mark))
#real_mark = np.zeros_like(real_mark)
#print(real_mark.sum())


#EMBEDDING AND DETECTION FUNCTIONS
def altEmbedding(image, mark_size=mark_size, beta=beta, v='multiplicative'):
    mark = np.random.uniform(0.0, 1.0, n_blocks**2)
    global real_mark
    mark[:mark_size]= real_mark.copy()
    mark[mark_size*mark_repetitions:] = 0
    for i in range(mark_size):
        for j in range(1,mark_repetitions):
            mark[j*mark_size+i]=mark[i]
    mark = mark.reshape((n_blocks,n_blocks))
    watermarked = image.copy()
    watermarked += beta*mark
    mark = mark.reshape(n_blocks**2)
    return mark[:mark_size], watermarked


def altDetection(image, watermarked, beta=beta, mark_size=mark_size, v='multiplicative'):
    global real_mark
    extended_mark = ((watermarked-image)/beta).reshape(n_blocks**2)
    extended_mark = np.clip(extended_mark,0,1)
    for i in range(mark_size):
        for j in range(1,mark_repetitions):
            #if (np.random.randint(0,25))<1: print("The right mark is {}, our bits are {} and {}".format(real_mark[i],extended_mark[i],extended_mark[i+mark_size]))
            extended_mark[i]+=extended_mark[j*mark_size+i]
    return (extended_mark/mark_repetitions)[:mark_size]


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


def embedding(image, mark_size=mark_size, beta=beta, v='additive'):

    blocksDC_image = image_to_blocksDC(image)

    mark, wd_blocksDC_image = altEmbedding(blocksDC_image,mark_size,beta,v)

    watermarked = change_blocksDC(image,wd_blocksDC_image)

    return mark, watermarked


def detection(image, watermarked, beta=beta, mark_size=mark_size, v='additive'):
    blocksDC_image = image_to_blocksDC(image)
    blocksDC_watermarked = image_to_blocksDC(watermarked)
    w_ex = altDetection(blocksDC_image,blocksDC_watermarked, beta, mark_size,v)
    return w_ex




















#AN EXAMPLE OF EMBEDDING

lena = Image.open('lena.bmp')
lena = np.asarray(Image.open('lena.bmp'), dtype=np.uint8)


mark, wd = embedding(lena, mark_size,beta,v)

#print('wPSNR: %.2fdB' % wpsnr(lena, wd))

w_ex = detection(lena, wd, beta, mark_size,v)

#print(similarity(mark, w_ex))



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
      imageList.append(np.asarray(Image.open('sample-images/{num:04d}.bmp'.format(num = i)), dtype=np.uint8))

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
        mark, watermarked = embedding(im, mark_size, beta, v)
        mean_wpsnr.append(wpsnr(im,watermarked))
        count = 0
        for j in range(200):
            for i in range(256):
                if im[i,j] == 255:
                    count +=1
        print(mean_wpsnr[-1], count)
        #show(im)
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
    print("average wPSNR: ", np.mean(mean_wpsnr))
    return scores, labels


scores, labels = create_scores_labels(10)

compute_roc(scores, labels)