from all_useful_things import *
import importlib

us = 'pixel'
group_names = ['ef26420c', 'you_shall_not_mark', 'blitz', 'omega', 'howimetyourmark', 'weusedlsb', 'thebavarians', 'theyarethesamepicture', 'dinkleberg', 'failedfouriertransform']

working_on = group_names[0] #change this if you want to work on other groups
working_on = us# only to test the script
print('Working on {}\n'.format(working_on))

#DETECTION FUNCTION
mod = importlib.import_module('detection_{}'.format(working_on))
detection = mod.detection

#IMAGES Here you need to add manually the names pf the images
image_names = ['buildings', '0001', '0002']
print('Selected images are: {}\n'.format(image_names))

#original images
o0 = cv2.imread('{}.bmp'.format(image_names[0]), 0)
#o1 = cv2.imread('{}.bmp'.format(image_names[1]), 0)
#o2 = cv2.imread('{}.bmp'.format(image_names[2]), 0)

#watermarked images
#w0 = cv2.imread('{}_{}.bmp'.format(working_on, image_names[0]), 0)
#w1 = cv2.imread('{}_{}.bmp'.format(working_on, image_names[1]), 0)
#w2 = cv2.imread('{}_{}.bmp'.format(working_on, image_names[2]), 0)

#attacked images
#a0 = w0.copy()
#a1 = w1.copy()
#a2 = w2.copy()



"""     here we do attack    """




#saving attacked images
#cv2.imwrite("{}_{}_{}.bmp".format(us,working_on,image_names[0]),a0)
#cv2.imwrite("{}_{}_{}.bmp".format(us,working_on,image_names[1]),a1)
#cv2.imwrite("{}_{}_{}.bmp".format(us,working_on,image_names[2]),a2)

#verify your attacks results
#print(detection("{}.bmp".format(image_names[0]), '{}_{}.bmp'.format(working_on, image_names[0])   ,  "{}_{}_{}.bmp".format(us,working_on,image_names[0]) ))
#print(detection("{}.bmp".format(image_names[1]), '{}_{}.bmp'.format(working_on, image_names[1])   ,  "{}_{}_{}.bmp".format(us,working_on,image_names[1]) ))
#print(detection("{}.bmp".format(image_names[2]), '{}_{}.bmp'.format(working_on, image_names[2])   ,  "{}_{}_{}.bmp".format(us,working_on,image_names[2]) ))


