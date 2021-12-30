# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:32:23 2021

@author: Jerry Yin
"""
#%% b)
import numpy as np
import matplotlib.pyplot as plt



def Euler(h):
    t_lis = []
    y_lis = []
    for t in np.arange(0,6,h):
        if t == 0:
            t_lis.append(t)
            y = 1
            y_lis.append(y)
        else:          
            diff_y = 2 - np.exp(-4 * t_lis[-1]) - (2 * y)
            t_lis.append(t)
            y_new = y + h * diff_y
            y_lis.append(y_new)
            y = y_new
    
    return t_lis, y_lis


T, YY = Euler(1)
X = np.linspace(0,5,1000)
Y = (0.5 * np.exp(-4*X)) - (0.5 * np.exp(-2 * X)) + 1
plt.figure()
plt.plot(T,YY)
plt.plot(X,Y)
plt.xlabel('t'), plt.ylabel('y')
plt.xlim(0,5)
plt.legend(['Euler approximated h=1','exact function'], loc=2)
plt.title('Comparison')
#%% c)

H = [0.1, 0.05, 0.01, 0.005, 0.001]
x= 1
plt.figure()
for h in H:
    tt, yy = Euler(h)
    plt.subplot(1,5,x)
    plt.plot(tt, yy)
    plt.xlabel('t'), plt.ylabel('y')
    plt.xlim(0,5)
    plt.title('approximated function \n h=%.3f' % h)
    x += 1

