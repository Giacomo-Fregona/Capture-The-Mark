from utility import *
import numpy as np

def diffwpsnr(diff):
    difference = diff/255.0
    same = not np.any(difference)
    if same is True:
      return 9999999
    csf = np.genfromtxt('csf.csv', delimiter=',')
    ew = convolve2d(difference, np.rot90(csf,2), mode='valid')
    decibels = 20.0*np.log10(1.0/sqrt(np.mean(np.mean(ew**2))))
    return decibels

stored=[]
pos = [[i,j] for i in range(8) for j in range (8)]
w = 0

num_ones= 50
for j in range(1000):
    image = np.zeros((88,88))
    pos_ones=[]
    for i in range(num_ones):
        r = pos[np.random.randint(64)]
        while(r in pos_ones):
            r = pos[np.random.randint(64)]
        pos_ones.append(r)
    for p in pos_ones:
        image[p[0]+40,p[1]+40] += 1
    val = diffwpsnr(image)
    if w<val:
        print(val)
        w = max(w,val)
        stored.append(image[40:48,40:48].copy())
print(w)


ref = np.uint8(100*np.ones((512, 512)))
for im in stored:
    show(im)
    IM = ref.copy()
    for i in range(512):
        ii = np.random.randint(28,484)
        jj = np.random.randint(28,484)
        IM[ii:(ii+8) , jj : (jj+8)] = IM[ii:(ii+8) , jj : (jj+8)] + im #* (-1+2*np.random.randint(0,1))
    show(IM)
    print(wpsnr(ref,IM))



""""
count = 0
for j in range(1):
    B = np.uint8([[np.random.randint(256) for i in range(8)] for i in range(8)])

    #B = IDCT(DCT(B))

    #show(B)

    D = DCT(B)
    for k in range(20):
        D [0,0]+=1
        print("k with value {}".format(k))
        BB = np.uint8(IDCT(D))

        if (BB-B).sum()> 0:
            print("\n\n\n{}".format((BB-B).sum()))
            #print(B, "\n\n")
            #print(D, "\n\n")
            #print(BB, "\n\nCon differenza:")
            print(BB-B)
            #show(B)
            count+=1

"""

#print (count)