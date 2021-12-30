# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 13:26:58 2021

@author: Jerry Yin
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

# read image
img = cv2.imread('lenaNoise.png', 0)
#plt.imshow(img,plt.cm.gray)
height, width = img.shape

# fft and shift
img_fourier = np.fft.fft2(img)
i_fftshift = np.fft.fftshift(img_fourier)
img_fftshift_for_plot = np.log(abs(i_fftshift))

# create mask
lis = [10, 20, 40, 60, 70, 80, height]
mask_lis = []
x = 1
fig = plt.figure()
for a in lis:
    mask = np.zeros(img.size, img.dtype).reshape(img.shape)
    for i in range(1, height):
        for j in range(1, width):
            # Creat squared mask
            if ((height - a) / 2) <= i <= ((height + a) /2) and ((width - a) / 2) <= j <= ((width + a) / 2):
                mask[i, j] = 1
            else:
                mask[i, j] = 0
    mask_lis.append(mask)
    # plot shifted frequency with mask    
    plt.subplot(3,len(lis),x),plt.imshow(mask_lis[x-1]+img_fftshift_for_plot, cmap = 'gray')
    plt.title('f=%d^2 with mask'% a), plt.xticks([]), plt.yticks([]) 
    # plot mask  
    plt.subplot(3,len(lis),x+len(lis)),plt.imshow(mask_lis[x-1], cmap = 'gray')
    plt.title('Mask of f=%d^2'% a), plt.xticks([]), plt.yticks([]) 
    # reconstruct image    
    re_image = np.fft.ifft2(np.fft.ifftshift(mask_lis[x-1] * i_fftshift))
    plt.subplot(3,len(lis),x+(2*len(lis))),plt.imshow(np.abs(re_image), cmap = 'gray')
    plt.title('Reconstructed Image f=%d^2'% a), plt.xticks([]), plt.yticks([]) 
    x += 1


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


