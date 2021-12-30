# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 11:35:29 2021

@author: jjerr
"""

import numpy as np
#from scipy.ndimage import filters
import matplotlib.pyplot as plt
import cv2
from skimage import img_as_float

def Gaussian_noise(img,sigma):
    ii = img_as_float(img)
    rng = np.random.default_rng()
    i_noisy = ii + sigma * rng.standard_normal(ii.shape)
    return i_noisy

def ROF_TV_denoise(im,U_init,epoch,ep,tau,tv_weight):
    """使用A. Chambolle(2005)在公式（11）中的计算步骤实现Rudin-Osher-Fatemi(ROF)去噪模型
        输入：含有噪声的输入图像（灰度图像）、U的初始值、TV正则项权值、步长、停业条件
        输出：去噪和去除纹理后的图像、纹理残留"""
    m,n = im.shape  #噪声图像的大小

    #初始化
    U = U_init
    #Px = im #对偶域的x分量
    #Py = im #对偶域为y分量
    Px = np.zeros((m,n))
    Py = np.zeros((m,n))
    error_old = 1
    itera = 0
    itera_lis = []
    error_lis = []
    for itera in range(epoch):
      
        Uold = U
        #原始变量的梯度
        LyU = np.vstack((U[1:,:],U[0,:])) #Left translation w.r.t. the y-direction
        LxU = np.hstack((U[:,1:],U.take([0],axis=1))) #Left translation w.r.t. the x-direction
        GradUx = LxU-U #x-component of U's gradient
        GradUy = LyU-U #y-component of U's gradient

        #更新对偶变量
        PxNew = Px + (tau/tv_weight)*GradUx
        PyNew = Py + (tau/tv_weight)*GradUy
        NormNew = np.maximum(1,np.sqrt(PxNew**2+PyNew**2))

        Px = PxNew/NormNew  #更新x分量（对偶）
        Py = PyNew/NormNew  #更新y分量（对偶）

        #更新原始变量
        RxPx = np.hstack((Px.take([-1],axis=1),Px[:,0:-1])) #Right x-translation of x-component
        RyPy = np.vstack((Py[-1,:],Py[0:-1,:])) #Right y-translation of y-component

        DivP = (Px-RxPx) + (Py-RyPy) #对偶域的散度
        U = im + tv_weight*DivP  #更新原始变量

        #更新误差
        error = (np.linalg.norm(U - Uold)/np.sqrt(n*m) + np.sqrt(PxNew**2+PyNew**2)/np.sqrt(n*m)).sum()/np.sqrt(n*m)
        if itera == 0:
            error_old = error
            itera_lis.append(itera)
            error_lis.append(error)
            print('Epoch = %d, Error = %f' % (itera, error))
            continue
        else:
            if np.abs(error - error_old) < ep:
                break
        error_old = error
        itera_lis.append(itera)
        error_lis.append(error)
        print('Epoch = %d, Error = %f' % (itera, error))
        
    T = im - U

    print('Total Epoches = %d' % itera)
    plt.figure()
    plt.plot(itera_lis, error_lis)
    plt.xlabel('Epoch'), plt.ylabel('Error')
    plt.title('Epoch vs Error (\u03C3=%.2f)' % sig)
    
    return U,T


img = cv2.imread('D:\\OneDrive - University of Virginia\\Courses\\ECE6501 Machine Learning in Image Analysis (Miaomiao Zhang)\\PS1\\PS1\\cameraman.png',0)
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
    #plt.figure()
    #for x in range(len(figg_lis)):
        #plt.subplot(3,1,x+1),plt.imshow(figg_lis[x], cmap='gray')
#plt.figure(4)
#plt.imshow(np.asarray(img_denoise[1]))



#%%












