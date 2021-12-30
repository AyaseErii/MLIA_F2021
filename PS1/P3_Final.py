# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 21:07:57 2021

@author: jjerr
"""

import numpy as np
#from scipy.ndimage import filters
import matplotlib.pyplot as plt
import cv2
from skimage import img_as_float
from skimage import io
from scipy.linalg import blas


def Gaussian_noise(img,sigma):
    ii = img_as_float(img)
    rng = np.random.default_rng()
    i_noisy = ii + sigma * rng.standard_normal(ii.shape)
    return i_noisy


def axpy(a, x, y):
    """Sets y = a*x + y and returns y."""
    shape = x.shape
    x, y = x.reshape(-1), y.reshape(-1)
    return blas.saxpy(x, y, a=a).reshape(shape)

axpy(1, np.array(0), np.array(0))

EPS = np.finfo(np.float64).eps

def tv_norm(x):
    """Computes the total variation norm and its gradient. From jcjohnson/cnn-vis."""
    x_diff = x - np.roll(x, -1, axis=1)
    y_diff = x - np.roll(x, -1, axis=0)
    grad_norm2 = x_diff**2 + y_diff**2 + EPS
    norm = np.sum(np.sqrt(grad_norm2))
    dgrad_norm = 0.5 / np.sqrt(grad_norm2)
    dx_diff = 2 * x_diff * dgrad_norm
    dy_diff = 2 * y_diff * dgrad_norm
    grad = dx_diff + dy_diff
    grad[:, 1:, :] -= dx_diff[:, :-1, :]
    grad[1:, :, :] -= dy_diff[:-1, :, :]
    #grad[:, :1] -= dx_diff[:, :-1]
    #grad[1:, :] -= dy_diff[:-1, :]
    return norm, grad


def l2_norm(x):
    """Computes 1/2 the square of the L2-norm and its gradient."""
    return np.sum(x**2) / 2, x


def main_func():
    img = io.imread('E:\\OneDrive - University of Virginia\\Courses\\ECE6501 Machine Learning in Image Analysis (Miaomiao Zhang)\\PS1\\PS1\\cameraman.png',0)
    img_noisy = Gaussian_noise(img, 0.1)
    orig_img = img.copy()        


    def opfunc(img, lmbda):
        tv_loss, tv_grad = tv_norm(img)
        l2_loss, l2_grad = l2_norm(img - orig_img)
        loss = tv_loss + l2_loss/lmbda
        grad = tv_grad + l2_grad/lmbda
        return loss, grad

    step_size = 1
    last_loss = 1
    steps = 0
    print('Optimizing using gradient descent.')
    while True:
        steps += 1
        loss, grad = opfunc(img_noisy,100)
        print('step:', steps, 'loss:', loss)
        if loss > last_loss:
            break
    last_loss = loss
    axpy(-step_size, grad, img)

    print('iterations', steps)        
        
        
if __name__ == "__main__":
    main_func()       
        
        
#%%        
sig_lis = [0.01, 0.05, 0.1]

if __name__ == "__main__":
    
    for sig in sig_lis:
        img_noisy =  Gaussian_noise(img,sig)
        #img_denoise, i_residual = ROF_TV_denoise(img,img,epoch=70,ep=2*(10**-4),tau=0.125,tv_weight=100)
        img_denoise, i_residual = ROF_TV_denoise(img_noisy,img_noisy,60,0.1,0.13,100)
        figg = plt.figure()
        plt.subplot(1,3,1),plt.imshow(img, cmap='gray')
        plt.title('Originial')
        plt.subplot(1,3,2),plt.imshow(img_noisy, cmap='gray')
        plt.title('Noise Image (\u03C3=%.2f)' % sig)
        plt.subplot(1,3,3),plt.imshow(np.asarray(img_denoise),cmap='gray')
        plt.title('TV Denoisy Image (\u03C3=%.2f)' % sig)       
        
        
        
        
        
        
        
        
        
        