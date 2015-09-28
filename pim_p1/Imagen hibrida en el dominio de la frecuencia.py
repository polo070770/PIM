
# coding: utf-8

# In[1]:

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

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


# In[2]:




# In[3]:

#Gaussian filter
fftsize = 1024
sigma = 9
SZ = sigma * 6 + 1
[xx,yy]=np.meshgrid(np.linspace(-4,4,SZ),np.linspace(-4,4,SZ))
gaussian = np.exp(-0.5*(xx*xx+yy*yy))
fil_fft = fftpack.fft2(gaussian/np.sum(gaussian), (fftsize, fftsize)) 


# In[4]:

#FOURIER SPACE CONVOLUTION 3 CHANNEL
def convolution_fourier_RGB(img, fil_fft, fftsize):

    channelR = np.zeros((img.shape[0],img.shape[1]), 'double')
    channelG = np.zeros((img.shape[0],img.shape[1]), 'double')
    channelB = np.zeros((img.shape[0],img.shape[1]), 'double')

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            channelR[x,y] = img[x,y][0]
            channelG[x,y] = img[x,y][1]
            channelB[x,y] = img[x,y][2]
        
    matrixR_fft = fftpack.fft2(channelR, (fftsize, fftsize))
    matrixG_fft = fftpack.fft2(channelG, (fftsize, fftsize))
    matrixB_fft = fftpack.fft2(channelB, (fftsize, fftsize))
    
    matrixR_fil_fft = matrixR_fft * fil_fft;
    matrixG_fil_fft = matrixG_fft * fil_fft;
    matrixB_fil_fft = matrixB_fft * fil_fft;

    matrixR_fil = np.real(fftpack.ifft2(matrixR_fil_fft))
    matrixG_fil = np.real(fftpack.ifft2(matrixG_fil_fft))
    matrixB_fil = np.real(fftpack.ifft2(matrixB_fil_fft))
    
    img_fil = np.zeros((matrixR_fil.shape[0], matrixR_fil.shape[1], 3), 'double')

    for x in range(matrixR_fil.shape[0]):
        for y in range(matrixR_fil.shape[1]):
            img_fil[x,y,0] = matrixR_fil[x,y]
            img_fil[x,y,1] = matrixG_fil[x,y]
            img_fil[x,y,2] = matrixB_fil[x,y]
            
    return img_fil


# In[5]:

#LOW PASS IMG1 
np_img1 = np.array(img1)
img1_filtered_LP = convolution_fourier_RGB(np_img1, fil_fft, fftsize)
hs=np.floor(SZ/2.)
img1_filtered_LP = img1_filtered_LP[hs:np_img1.shape[0]+hs, hs:np_img1.shape[1]+hs]
img1_filtered_LP /= img1_filtered_LP.max()


# In[6]:

#HIGH PASS IMG2
np_img2 = np.array(img2)
img2_filtered_LP = convolution_fourier_RGB(np_img2, fil_fft, fftsize)
hs=np.floor(SZ/2.)
img2_filtered_LP = img2_filtered_LP[hs:np_img2.shape[0]+hs, hs:np_img2.shape[1]+hs]
img2_filtered_HP = np_img2 - img2_filtered_LP

img2_filtered_HP -= np.amin(img2_filtered_HP)
img2_filtered_HP /= img2_filtered_HP.max()


# In[7]:

#HYBRID IMG
img_hybrid = (img1_filtered_LP + img2_filtered_HP) / 2.


# In[8]:

#SHOWING IMGS
plt.figure(3)
plt.subplot(121)
plt.imshow(img1)
plt.title('original img1', size=16)
plt.subplot(122)
plt.imshow(img1_filtered_LP)
plt.title('LP img1', size=16)
plt.gcf().set_size_inches((14,14))


plt.figure(4)
plt.subplot(121)
plt.title('original img2', size=16)
plt.imshow(img2)
plt.subplot(122)
plt.imshow(img2_filtered_HP)
plt.title('HP img2', size=16)
plt.gcf().set_size_inches((14,14))


plt.figure(5)
plt.imshow(img_hybrid)
plt.title('hybrid image', size=16)
plt.gcf().set_size_inches((14,14))


# In[8]:



