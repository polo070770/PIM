
# coding: utf-8

## Imagen hibrida en el dominio del espacio 

# In[1]:

from PIL import Image
import math
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from scipy import signal as sg

get_ipython().magic(u'matplotlib inline')


# In[2]:

#LOAD IMAGES 
img1 = Image.open("human.png")
img2 = Image.open("cat.png")

#RESIZE 
size1=int(round(img1.size[0]*(4/3.))); 
size2=int(round(img1.size[1]*(4/3.))); 
img1=img1.resize((size1,size2), Image.ANTIALIAS)

#CROP 
left = 50 
top = 50 
right = 51+img2.size[0]-1 
bottom = 51+img2.size[1]-1 
img1=img1.crop((left, top, right, bottom))

# VISUALIZATION
plt.figure(1) 
plt.subplot(131) 
imgplot1=plt.imshow(img1) 
imgplot1.set_cmap('gray') 
plt.subplot(132) 
imgplot1=plt.imshow(img2) 
imgplot1.set_cmap('gray')


# In[3]:

#GAUSSIAN FUNCTION
def gaussian(x, y, size, sigma):
    return (1./ (2. * math.pi * sigma * sigma)) * math.exp(- (math.pow(x-(size/2.),2) + math.pow(y-(size/2.),2)) / (2. * sigma * sigma))

#GAUSSIAN FILTER FUNCTION
def gaussian_filter(sigma):
    size = 6 * sigma + 1
    gauss_matrix = np.zeros( (size, size) )
    
    for x in range(size):
        for y in range(size):
            gauss_matrix[x][y] = gaussian(x, y, size, sigma)
    return gauss_matrix


# In[4]:

#CONVOLUTION 3 CHANNEL FUNCTION 
def convolucion_RGB(img, kernel):

    channelR = np.zeros((img.shape[0],img.shape[1]), 'uint8')
    channelG = np.zeros((img.shape[0],img.shape[1]), 'uint8')
    channelB = np.zeros((img.shape[0],img.shape[1]), 'uint8')

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            channelR[x,y] = img[x,y][0]
            channelG[x,y] = img[x,y][1]
            channelB[x,y] = img[x,y][2]
        
    matrixR = sg.convolve2d(channelR, kernel, "same")
    matrixG = sg.convolve2d(channelG, kernel, "same")
    matrixB = sg.convolve2d(channelB, kernel, "same")

    img_conv = np.zeros((matrixR.shape[0], matrixR.shape[1], 3), 'double')

    for x in range(matrixR.shape[0]):
        for y in range(matrixR.shape[1]):
            img_conv[x,y,0] = matrixR[x,y]
            img_conv[x,y,1] = matrixG[x,y]
            img_conv[x,y,2] = matrixB[x,y]
        
    return img_conv


# In[5]:

np_img1 = np.array(img1).astype('double')
np_img2 = np.array(img2).astype('double')

#GAUSS KERNEL
kernel = gaussian_filter(9)
kernel = kernel/sum(sum(kernel)) #Gauss normalizado

#LOW PASS IMG1
img1_LowPass = convolucion_RGB(np_img1, kernel)
img1_LowPass /= img1_LowPass.max() #Img LP normalizado

img2_LowPass = convolucion_RGB(np_img2, kernel)
img2_HighPass = np_img2 - img2_LowPass

#HIGH PASS IMG2
img2_HighPass = img2_HighPass - np.amin(img2_HighPass)
img2_HighPass /= img2_HighPass.max()

#HYBRID IMG
hybrid_img = (img1_LowPass + img2_HighPass) / 2.


# In[6]:

#SHOWING IMGS
plt.figure(1)
plt.subplot(121)
plt.imshow(img1)
plt.title('original img1', size=16)
plt.subplot(122)
plt.imshow(img1_LowPass)
plt.title('LP img1', size=16)
plt.gcf().set_size_inches((14,14))


plt.figure(2)
plt.subplot(121)
plt.title('original img2', size=16)
plt.imshow(img2)
plt.subplot(122)
plt.imshow(img2_HighPass)
plt.title('HP img2', size=16)
plt.gcf().set_size_inches((14,14))


plt.figure(3)
plt.imshow(hybrid_img)
plt.title('hybrid image', size=16)
plt.gcf().set_size_inches((14,14))

