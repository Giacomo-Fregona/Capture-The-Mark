from all_useful_things import *
import importlib

us = 'pixel'
group_names = ['ef26420c', 'youshallnotmark', 'blitz', 'omega', 'howimetyourmark', 'weusedlsb', 'thebavarians', 'theyarethesamepicture', 'dinkleberg', 'failedfouriertransform']

working_on = group_names[2] #change this if you want to work on other groups
#working_on = us# only to test the script
print('Working on {}\n'.format(working_on))

#DETECTION FUNCTION


#import detection_failedfouriertransform
#detection = detection_failedfouriertransform.detection



import detection_pixel
detection = detection_pixel.detection
import embedding_pixel
embedding = embedding_pixel.embedding


#IMAGES Here you need to add manually the names pf the images
image_names = ['buildings', 'tree', 'rollercoaster']
print('Selected images are: {}\n'.format(image_names))

#original images
o0 = cv2.imread('{}.bmp'.format(image_names[0]), 0)
o1 = cv2.imread('{}.bmp'.format(image_names[1]), 0)
o2 = cv2.imread('{}.bmp'.format(image_names[2]), 0)

#watermarked images
w0 = cv2.imread('{}_{}.bmp'.format(working_on, image_names[0]), 0)
w1 = cv2.imread('{}_{}.bmp'.format(working_on, image_names[1]), 0)
w2 = cv2.imread('{}_{}.bmp'.format(working_on, image_names[2]), 0)


#attacked images
a0 = w0.copy()
a1 = w1.copy()
a2 = w2.copy()

w = embedding('{}.bmp'.format(image_names[0]))[1]


plt.figure(figsize=(15, 6))
plt.subplot(131)
plt.title('Original', fontsize = 30)
plt.imshow(o0, cmap='gray')
plt.subplot(132)
plt.title('Watermarked', fontsize = 30)
plt.imshow(w, cmap='gray')
plt.subplot(133)
plt.title('Mask for the attack', fontsize = 30)
plt.imshow(w-o0, cmap='gray')
plt.show()






"""     here we do attack    """
i = 43

def attack(img, i):
    return blur(img, [0.5,0.5])

#def attack(img, i):
#    return awgn(img, 1.0, np.random.randint(1,100))

def attack(img, i):
    return jpeg_compression(img, i)
print("\n\nattacking with parameter {}".format(i))




a0 = borders(a0)
a1 = borders(a1)
a2 = borders(a2)


#saving attacked images
cv2.imwrite("{}_{}_{}.bmp".format(us,working_on,image_names[0]),a0)
cv2.imwrite("{}_{}_{}.bmp".format(us,working_on,image_names[1]),a1)
cv2.imwrite("{}_{}_{}.bmp".format(us,working_on,image_names[2]),a2)
#verify your attacks results
print("\nimage = {}".format(image_names[0]))
print(detection("{}.bmp".format(image_names[0]), '{}_{}.bmp'.format(working_on, image_names[0])   ,  "{}_{}_{}.bmp".format(us,working_on,image_names[0]) ))

print("\nimage = {}".format(image_names[1]))
print(detection("{}.bmp".format(image_names[1]), '{}_{}.bmp'.format(working_on, image_names[1])   ,  "{}_{}_{}.bmp".format(us,working_on,image_names[1]) ))

print("\nimage = {}".format(image_names[2]))
print(detection("{}.bmp".format(image_names[2]), '{}_{}.bmp'.format(working_on, image_names[2])   ,  "{}_{}_{}.bmp".format(us,working_on,image_names[2]) ))


