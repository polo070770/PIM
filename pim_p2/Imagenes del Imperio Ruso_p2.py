
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import math
import scipy
from scipy import ndimage, signal
from scipy.misc import imresize

get_ipython().magic(u'matplotlib inline')


# In[2]:

#LOAD IMAGES
img = Image.open("00029u.png")

#WARNING SIZE AND SHAPE INDICES ARE UPSIDE DOWN
size1_size=int(round(img.size[1]/10))
size2_size=int(round(img.size[0]/10))

img = imresize(img, ( size1_size,size2_size),interp='bilinear').astype('float')
img_np = np.array(img)

x1 = 20; y1 = 20;
x2 = 25; y2 = 333;
x3 = 24; y3 = 647;
w = 336; h = 302;

im1 = img_np[y1-2:y1+h-2,x1-1:x1+w-1];
im2 = img_np[y2-2:y2+h-2,x2-1:x2+w-1];
im3 = img_np[y3-2:y3+h-2,x3-1:x3+w-1];

I1 = 256*im1.astype('double')/im1.max();
I2 = 256*im2.astype('double')/im2.max();
I3 = 256*im3.astype('double')/im3.max();

#Union of the 3 channels to get the image
RGB=np.zeros([h,w,3],dtype=type(img_np[0,0]))
RGB[:,:,0]=I3
RGB[:,:,1]=I2
RGB[:,:,2]=I1

#Get the 3 channel by separate
R = np.copy(RGB[:,:,0])
G = np.copy(RGB[:,:,1])
B = np.copy(RGB[:,:,2])


# In[6]:

B.shape


# In[ ]:



