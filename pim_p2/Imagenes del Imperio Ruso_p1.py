
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


# In[5]:

def maximIdx(matrix):
    """
        Function that finds the max value index.
    """
    maxim = -float('inf')
    for row in matrix:
        maxim_local = max(row)
        if maxim_local > maxim:
            maxim = maxim_local
    return np.where(maxim == matrix)


# In[6]:

def normalizeIt(matrix):
    """
        Function that normalize a matrix and then return it.
    """
    matrix -= np.min(matrix)
    matrix /= np.max(matrix)
    matrix -= np.max(matrix)
    return matrix


# In[7]:

def cutIt(matrix, x, y, width, height):
    """
        Function that returns a IN matrix copy crop it with the IN values
    """
    return normalizeIt(np.copy(matrix[x:x + height, y:y + width]))


# In[8]:

def cross_correlate(search_img, template_img):
    """
        Function that returns the correlation of two matrix.
    """
    return signal.correlate2d(search_img.astype('float'), template_img.astype('float'), mode='full', boundary='fill', fillvalue=0)


# In[9]:

def img_displacement(matrix, cord_0):
    """
        Function that returns the displacement of matrix max value coordinates and 
        initial coordinates
        
    """
    #The max value coordinates
    coord = maximIdx(matrix)

    print "\nMax value coordenates: ", coord[0], coord[1]

    #Displacement
    dx = cord_0[0] - coord[0]
    dy = cord_0[1] - coord[1]

    print "Displacement: ", dx, dy
    
    return dx, dy


# In[10]:

#Channel R pixel(x,y) what i'm going to match with the others channels
x0 = 140
y0 = 100


# In[11]:

#BLOCK TO SEARCH
x_B = x0 + 20
y_B = y0 + 15

_size_width = 40
_size_height = 20

search_R = cutIt(R, x_B, y_B, _size_width, _size_height)

#TEMPLATE FOR EVERY CHANNEL
x_T = x0 + 10
y_T = y0 + 10

size_width = 55
size_height = 40

template_R = cutIt(R, x_T, y_T, size_width, size_height)
template_G = cutIt(G, x_T, y_T, size_width, size_height)
template_B = cutIt(B, x_T, y_T, size_width, size_height)


# In[12]:

#Autocorrelation with the search_R and the template_R, to find the pattern at the template and know the position relative.
R_autocorr = cross_correlate(search_R, template_R)
position_relative_R = maximIdx(R_autocorr)

#Cross correlation with the image search_R and template_G
R_G_corr = cross_correlate(search_R, template_G)
displ_coord_G = img_displacement(R_G_corr, position_relative_R)

#Cross correlation with the image search_R and template_B
R_B_corr = cross_correlate(search_R, template_B)
displ_coord_B = img_displacement(R_B_corr, position_relative_R)


# In[14]:

img_align = np.zeros((size_height, size_width, 3))

img_align[:,:,0] = R[x_T : x_T + size_height, y_T:y_T + size_width]

img_align[:,:,1] = G[x_T + displ_coord_G[0] : x_T + displ_coord_G[0] + size_height,
                     y_T + displ_coord_G[1] : y_T + displ_coord_G[1] + size_width]

img_align[:,:,2] = B[x_T + displ_coord_B[0] : x_T + displ_coord_B[0] + size_height,
                     y_T + displ_coord_B[1] : y_T + displ_coord_B[1] + size_width]

img_align = normalizeIt(img_align)


# In[15]:

plt.imshow(img_align)


# In[ ]:



