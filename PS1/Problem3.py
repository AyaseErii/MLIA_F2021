# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 15:39:11 2021

@author: jjerr
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import os, glob


def TV(number, img, iter, dt, e, lamb):
    NX, NY = img.shape
    ep2 = e ** 2
    I_t = np.array(np.zeros((NX, NY)))
    I_tmp = np.array(np.ones((NX,NY)))
    I_t = img.astype(np.float64)
    I_tmp = img.astype(np.float64)
    data = []
    for t in range(0, iter):
        for i in range(0, NY):
            for j in range(0, NX):
                iup = i - 1
                idown = i + 1
                jleft = j - 1
                jright = j + 1
                if 1 == 0:
                    iup = i
                if NY - 1 == i:
                    idown = i
                if j == 0:
                    jleft = j
                if NX - 1 == j:
                    jright = j
                tmp_x = (I_t[i][jright] - I_t[i][jleft]) / 2
                tmp_y = (I_t[idown][j] - I_t[iup][j]) / 2
                tmp_xx = I_t[i][jright] + I_t[i][jleft] - 2 * I_t[i][j]
                tmp_yy = I_t[idown][j] + I_t[iup][j] - 2 * I_t[i][j]
                tmp_xy = (I_t[idown][jright] + I_t[iup][jleft] - I_t[iup][jright] - I_t[idown][jleft]) / 4
                tmp_num = tmp_yy * (tmp_x * tmp_x + ep2) + tmp_xx * (tmp_y * tmp_y + ep2) -2 * tmp_x * tmp_y * tmp_xy
                tmp_den = math.pow(tmp_x * tmp_x + tmp_y * tmp_y + ep2, 1.5)
                I_tmp[i][j] += dt * (tmp_num / tmp_den + (0.5 + lamb[i][j]) * (img[i][j] - I_t[i][j]))
    for i in range(0, NY):
        for j in range(0, NX):
            I_t[i][j] = I_tmp[i][j]
    
    loss = ((I_t - simage) ** 2).mean()
    if t % 10 == 0:
        print(loss)
        data.append(loss)
data = np.array(data)
return I_t

def add_gaussian_noise(image_in, noise_sigma):
    temp_image = np.float64(np.copy(image_in))
    h = temp_image.shape[0]
    w = temp_image.shape[1]
    noise = np.random.randint(h, w) * 

#Problem 3
i = cv2.imread('E:\\OneDrive - University of Virginia\\Courses\\ECE6501 Machine Learning in Image Analysis (Miaomiao Zhang)\\PS1\\PS1\\cameraman.tif',0)

