# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 11:35:29 2021

@author: jjerr
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import img_as_float

def Gaussian_noise(im,sigma):
    i_source = img_as_float(im)
    rng = np.random.default_rng()
    i_noisy = i_source + sigma * rng.standard_normal(i_source.shape)
    return i_noisy

def GD_TVROF_denoise(im,U0,epoch,ep,t,totalv_w): #im:source img, U0:noisy, epoch: iteration number, ep: epsilon, t: step control factor, totalv_w: tv weight
    hei,wid = im.shape  #shape of the image   
    U = U0 #Initializaton
    x = np.zeros((hei,wid)) # set x values as np.zeros: dual space at X direction
    y = np.zeros((hei,wid)) # set y values as np.zeros: dual space at Y direction
    error_old = 1 #For make comparison with the current error
    itera_lis = []
    error_lis = []
    for itera in range(epoch):
        U_od = U #Pass the U to the previous U for minusing
        Leftx = np.hstack((U[:,1:],U.take([0],axis=1))) #Left translation on the x-direction
        Lefty= np.vstack((U[1:,:],U[0,:])) #Left translation on the y-direction        
        Difxx = Leftx-U #x-component of U's gradient
        Difyy = Lefty-U #y-component of U's gradient

        #Update update the variables at dual space
        xnw = x + (t/totalv_w)*Difxx
        ynw = y + (t/totalv_w)*Difyy
        Normnw = np.maximum(1,np.sqrt(xnw**2+ynw**2))
        x = Difxx/Normnw  #Update the dual space at X direction
        y = Difyy/Normnw  #Update the dual space at Y direction
        
        #Update initial variables
        Rightxx = np.hstack((x.take([-1],axis=1),x[:,0:-1])) #Right X-translation of X-component
        Rightyy = np.vstack((y[-1,:],y[0:-1,:])) #Right Y-translation of Y-component

        Divergence = (x-Rightxx) + (y-Rightyy) #Divergence
        Stp = t*(np.exp(-itera*2/epoch)+0.1) #Set the step based on epoches and t
        U = U_od-Stp*((U_od-im) + totalv_w*Divergence) # Gradient descent

        #update the error       
        error = (np.linalg.norm(U - U_od)/np.sqrt(hei*wid) + np.sqrt(xnw**2+ynw**2)/np.sqrt(hei*wid)).sum()/(np.sqrt(hei*wid))

        if itera == 0:
            error_old = error
            continue
        else:
            if np.abs(error - error_old) < ep:
                break
        error_old = error
        itera_lis.append(itera)
        error_lis.append(error)
        print('Epoch = %d, Error = %f' % (itera, error))
        
    
    print('Total Epoches = %d' % itera)        
    plt.figure()
    plt.plot(itera_lis, error_lis)
    plt.xlabel('Epoch'), plt.ylabel('Error')
    plt.title('Epoch vs Error (\u03C3=%.2f)' % sig)
    
    return U #return denoisy image


img = cv2.imread('cameraman.png',0) #input: source image

# set different sigma for noising the source image
sig_lis = [0.01, 0.05, 0.1] 

# main function and plotting
i = 1
if __name__ == "__main__":
    for sig in sig_lis:
        img_noisy =  Gaussian_noise(img,sig)
        img_denoise = GD_TVROF_denoise(img,img_noisy,1000,0.0001,0.00001,10)
        plt.figure(2)
        plt.subplot(3,3,i),plt.imshow(img, cmap='gray')
        plt.title('Originial')
        plt.subplot(3,3,i+1),plt.imshow(img_noisy, cmap='gray')
        plt.title('Noise Image (\u03C3=%.2f)' % sig)
        plt.subplot(3,3,i+2),plt.imshow(np.asarray(img_denoise),cmap='gray')
        plt.title('TV Denoisy Image (\u03C3=%.2f)' % sig)
        i += 3
        














