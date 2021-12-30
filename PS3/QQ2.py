# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 10:29:27 2021

@author: Jerry Yin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def pick_two_specific_digits(X_train,Y_train,digit_1,digit_2):
    df_X_train = pd.DataFrame() ## empty df for x
    df_Y_train = pd.DataFrame() ## empty df for y 
    index_list = []
    
    real_num_digit_1 = 0
    real_num_digit_2 = 0
    
    for ind in range(len(X_train)):
        if Y_train[0][ind] == digit_1:
            index_list.append(ind)
            real_num_digit_1 += 1
        if Y_train[0][ind] == digit_2:
            index_list.append(ind)
            real_num_digit_2 += 1
    
    X_picked = df_X_train.append([X_train.T[i] for i in index_list])
    Y_picked = df_Y_train.append([Y_train.T[i] for i in index_list])
    
    return X_picked.reset_index(drop=True), Y_picked.reset_index(drop=True), real_num_digit_1, real_num_digit_2


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
    s_eigvec = eigvec[:,sind]    
     
    return s_eigvec


def sigmoid(z):
    
    return np.exp(-z) / (1.0 + np.exp(-z))


def Gradient_Ascent(X,Y,sigma2,step,N):
    betas = np.zeros((X.shape[1]))
    summ = betas
    itera = 0
    itera_lis = []
    loss_lis = []
    m = X.shape[0]
    start = time.time()
    for itera in range(N):   
        itera_lis.append(itera)        
        for i in range(m):
            summ = summ + (Y[0][i] - 1 + sigmoid(np.dot(X[i],betas))) * X[i]
        G = summ - (1/sigma2) * betas    # loss
        loss_lis.append(np.abs(np.mean(G)))  # taking np.abs for plotting 
        betas_new = betas + step * G   # gradient ascent
        betas = betas_new
    t = time.time() - start
    betas_final = betas
    return itera_lis, loss_lis, betas_final, t


def classifier(X,Y,b):
    mmm = -np.dot(X,b)
    exp = np.exp(mmm)
    P_y = 1/(1 + exp) # P(y_i = 1)
    Y_hat_lis = []
    Y_hat = 0
    for p in P_y:
        if p >= 0.5:
            Y_hat = 1
            Y_hat_lis.append(Y_hat)
        if 0 <= p < 0.5:
            Y_hat = 0
            Y_hat_lis.append(Y_hat)
    return Y_hat_lis


def Accuracy(Y_hat,Y):
    count = 0
    Y = np.array(Y)
    E_lis = []
    for i in range(len(Y_hat)):
        E_lis.append(int(Y[i] - Y_hat[i]))
        if Y_hat[i] == Y[i]:
            count += 1
        else:
            continue
    accur = count / len(Y_hat)
    return accur,E_lis


# Read files. If it cannot be run, please change the path to absolute path, thanks!
X_train = pd.read_csv('trainX.csv', header=None)
Y_train = pd.read_csv('trainY.csv', header=None)

X_test = pd.read_csv('testX.csv', header=None)
Y_test = pd.read_csv('testY.csv', header=None)


#X_train = pd.read_csv('D:\\OneDrive - University of Virginia\\Courses\\ECE6501 Machine Learning in Image Analysis (Miaomiao Zhang)\\PS2\\PS2_Q2_mnist_for_python\\mnist for python\\trainX.csv', header=None)
#Y_train = pd.read_csv('D:\\OneDrive - University of Virginia\\Courses\\ECE6501 Machine Learning in Image Analysis (Miaomiao Zhang)\\PS2\\PS2_Q2_mnist_for_python\\mnist for python\\trainY.csv', header=None)

#X_test = pd.read_csv('D:\\OneDrive - University of Virginia\\Courses\\ECE6501 Machine Learning in Image Analysis (Miaomiao Zhang)\\PS2\\PS2_Q2_mnist_for_python\\mnist for python\\testX.csv', header=None)
#Y_test = pd.read_csv('D:\\OneDrive - University of Virginia\\Courses\\ECE6501 Machine Learning in Image Analysis (Miaomiao Zhang)\\PS2\\PS2_Q2_mnist_for_python\\mnist for python\\testY.csv', header=None)


X_train_0_1s, Y_train_0_1s, total_train_0s, total_train_1s = pick_two_specific_digits(X_train,Y_train,0,1) # pick 0s and 1s only

X_test_0_1s, Y_test_0_1s, total_test_0s, total_test_1s = pick_two_specific_digits(X_test,Y_test,0,1) # pick 0s and 1s only



X_train_PCA_eigvec = PCA_func(X_train_0_1s)
X_test_PCA_eigvec = PCA_func(X_test_0_1s)


PC_lis = [10,20,30]
t_lis = []
a_lis = []
plt.figure(1)
for i in range(len(PC_lis)):
    
    X_train_PCA_eigvec_new = X_train_PCA_eigvec.T[:PC_lis[i]] # PC-dimenstional eigvec for training set
    X_test_PCA_eigvec_new = X_test_PCA_eigvec.T[:PC_lis[i]] # PC-dimenstional eigvec for test set
    
    low_X_train_0_1s = np.linalg.multi_dot([X_train_0_1s, X_train_PCA_eigvec_new.T]) # Final low dimension training set
    low_X_test_0_1s = np.linalg.multi_dot([X_test_0_1s, X_test_PCA_eigvec_new.T]) # Final low dimension test set
        
    
    insert_1s_train = np.ones((low_X_train_0_1s.shape[0], 1))
    low_X_train_0_1s = np.hstack((insert_1s_train, low_X_train_0_1s)) # add a col of 1s to final low dimension training set
    
    insert_1s_test = np.ones((low_X_test_0_1s.shape[0], 1))
    low_X_test_0_1s = np.hstack((insert_1s_test, low_X_test_0_1s)) # add a col of 1s to final low dimension test set
    
    
    iterr, los, Beta, t = Gradient_Ascent(low_X_train_0_1s,Y_train_0_1s,sigma2=0.001,step=0.001,N=1000) # regression
    t_lis.append(t)
    
    Y_hats = classifier(low_X_test_0_1s, Y_test_0_1s,Beta)
    a, Expects = Accuracy(Y_hats,Y_test_0_1s)

    a_lis.append(a)
    wrong = len(Expects) - Expects.count(0)
    E1 = sum(Expects)/len(Expects)
    print('When PC=%d, the accuracy for 0 and 1 classification is %.6f' % (PC_lis[i],a))
    print('Predicted Y (Y_tilde) = %s' % Y_hats)
    print('E[|Y-Y_tilde|] = %f' % E1)
    print('Wrongly classified amount: %d' % wrong)
    print('Time consuming:%.2f s\n' % t)
    
    plt.subplot(1,len(PC_lis),i+1)
    plt.plot(iterr,los)
    plt.xlabel('Epoch'),plt.ylabel('Loss')
    plt.title('0 and 1 classification (PC=%d)' % PC_lis[i])
    plt.legend(['Accuracy is %.6f' % a], loc=1)


plt.figure(2)
plt.subplot(1,2,1),plt.plot(PC_lis,a_lis)
plt.xlabel('PCs'),plt.ylabel('Accuracy')
plt.xlim(5,35)
plt.title('PCs vs. Accuracy')


plt.subplot(1,2,2),plt.scatter(PC_lis,t_lis,c='r')
plt.xlabel('PCs'),plt.ylabel('Consumed Time (s)')
plt.xlim(5,35)
plt.title('PCs vs. Consumed Time')












