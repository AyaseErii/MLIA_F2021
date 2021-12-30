# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 14:37:42 2021

@author: Jerry Yin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 

# Build PCA function 
def PCA_func(D): 

    # Subtracting mean first
    D = np.array(D)
    D_avg = D - np.mean(D, axis=0)
     
    # Getting covariance matrix
    K = np.matmul(D_avg.T, D_avg)/(len(D_avg)-1)
     
    # Getting eigenvalues and eigenvectors
    eigval , eigvec = np.linalg.eigh(K)
     
    # Sorting eigenvalues and eigenvectors in descending order
    sind = eigval.argsort()[::-1]
    s_eigval = eigval[sind]
    s_eigvec = eigvec[:,sind]
    
    # Calculating the weight of a single feature value  
    total = sum(s_eigval)
    var = [(i/total)*100 for i in s_eigval]
    
    # Calculating the accumulated weight of feature values
    cum_var = np.cumsum(var)
     
    return s_eigval, cum_var, s_eigvec


D = pd.read_csv('D:\\OneDrive - University of Virginia\\Courses\\ECE6501 Machine Learning in Image Analysis (Miaomiao Zhang)\\PS2\\PS2_Q2_mnist_for_python\\mnist for python\\trainX.csv', header=None)


evalue, cumsum, s_evec = PCA_func(D)


# Plot all eigenvalues
plt.figure(1)
plt.plot(evalue)
plt.xlabel('Label of Principal Component'), plt.ylabel('Eigenvalue')
plt.title('All eigenvalues')

# Plot Number of Principal Components vs. Accumulated Data Variance
p = [] 
pp = []
for i in range(len(cumsum)):
    if cumsum[i] >= 90:
        p.append(i)
        pp.append(cumsum[i])
        break
xx = np.linspace(0, 800, 1000)
yy = np.linspace(90, 90, 1000)
plt.figure(2)
plt.plot(cumsum, 'y')
plt.plot(xx, yy, 'b--')
plt.scatter(p, pp)
plt.xlim(0,100)
plt.xticks([0,10,20,30,40,50,60,70,80,90,100])
plt.yticks([0,10,20,30,40,50,60,70,80,90,100])
plt.xlabel('Number of Principal Components'), plt.ylabel('Accumulated Data Variance (%)')
plt.legend(['Accumulated Data Variance','Accumulated Data Variance = 90%','(%d,%.3f)' % (p[0], pp[0])], loc=4)



# Plot first 10 eigenvectors
plt.figure(3)
for i in range(0,10):
    plt.subplot(2,5,i+1)
    plt.imshow((s_evec.T[i]).reshape(28,28), cmap='gray')       
    plt.title('Eigenvector %d' % (i+1))
    plt.xticks()
    plt.yticks()
plt.show()


