# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 18:13:48 2021

@author: Jerry Yin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# For 0 and 1 classification

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


def sigmoid(z):
    
    return np.exp(-z) / (1.0 + np.exp(-z))


def Gradient_Ascent(X,Y,sigma2,step,N):
    betas = np.zeros((X.shape[1]))
    summ = betas
    itera = 0
    itera_lis = []
    loss_lis = []
    m = X.shape[0]
    for itera in range(N):   
        itera_lis.append(itera)        
        for i in range(m):
            summ = summ + (Y[0][i] - 1 + sigmoid(np.dot(X[i],betas))) * X[i]
        G = summ - (1/sigma2) * betas    
        loss_lis.append(np.abs(np.mean(G)))   
        betas_new = betas + step * G   
        betas = betas_new

    betas_final = betas
    return itera_lis, loss_lis, betas_final


def classifier(X,Y,b):
    mmm = -np.dot(X,b)
    exp = np.exp(mmm)
    P_y = 1/(1 + exp)
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
X_train = pd.read_csv('E:\\OneDrive - University of Virginia\\Courses\\ECE6501 Machine Learning in Image Analysis (Miaomiao Zhang)\\PS2\\PS2_Q2_mnist_for_python\\mnist for python\\trainX.csv', header=None)
Y_train = pd.read_csv('E:\\OneDrive - University of Virginia\\Courses\\ECE6501 Machine Learning in Image Analysis (Miaomiao Zhang)\\PS2\\PS2_Q2_mnist_for_python\\mnist for python\\trainY.csv', header=None)

X_test = pd.read_csv('E:\\OneDrive - University of Virginia\\Courses\\ECE6501 Machine Learning in Image Analysis (Miaomiao Zhang)\\PS2\\PS2_Q2_mnist_for_python\\mnist for python\\testX.csv', header=None)
Y_test = pd.read_csv('E:\\OneDrive - University of Virginia\\Courses\\ECE6501 Machine Learning in Image Analysis (Miaomiao Zhang)\\PS2\\PS2_Q2_mnist_for_python\\mnist for python\\testY.csv', header=None)


X_train_0_1s, Y_train_0_1s, total_train_0s, total_train_1s = pick_two_specific_digits(X_train,Y_train,0,1) # pick 0s and 1s only
X_test_0_1s, Y_test_0_1s, total_test_0s, total_test_1s = pick_two_specific_digits(X_test,Y_test,0,1) # pick 0s and 1s only

insert_1s_train = np.ones((X_train_0_1s.shape[0], 1))
X_train_0_1s = np.hstack((insert_1s_train, X_train_0_1s))

insert_1s_test = np.ones((X_test_0_1s.shape[0], 1))
X_test_0_1s = np.hstack((insert_1s_test, X_test_0_1s))

#training for getting Beta
iterr, los, Beta = Gradient_Ascent(X_train_0_1s,Y_train_0_1s,sigma2=0.001,step=0.001,N=60)

Y_hats = classifier(X_test_0_1s, Y_test_0_1s,Beta)
a, Expectations = Accuracy(Y_hats,Y_test_0_1s)
wrong = len(Expectations) -  Expectations.count(0)
E1 = sum(Expectations)/len(Expectations)
print('Beta vector for 0 and 1 classification =\n%s' % Beta)
print('Accuracy for 0 and 1 classification is %.6f' % a)
print('Predicted Y (Y_tilde) = %s' % Y_hats)
print('Y-Y_tilde =',Expectations)
print('E[|Y-Y_tilde|]=',E1)
print('Wrongly classified amount: %d\n' % wrong)



plt.figure(1)
plt.plot(iterr,los)
plt.xlabel('Epoch'),plt.ylabel('Loss')
plt.title('0 and 1 classification')
plt.legend(['Accuracy is %.6f' % a], loc=1)











# For 6 and 8 classification
def Gradient_Ascent2(X,Y,sigma2,step,N): # for 6 and 8 only
    betas = np.zeros((X.shape[1]))
    summ = betas
    itera = 0
    itera_lis = []
    loss_lis = []
    m = X.shape[0]
    for itera in range(N):   
        itera_lis.append(itera)        
        for i in range(m):
            #summ = summ + (Y[0][i] - 1 + sigmoid(X[i] * betas)) * X[i]
            if Y[0][i] == 6:
                summ = summ + (Y[0][i] -6 - 1 + sigmoid(np.dot(X[i],betas))) * X[i]
            if Y[0][i] == 8:
                summ = summ + (Y[0][i] -7 - 1 + sigmoid(np.dot(X[i],betas))) * X[i]
        G = summ - (1/sigma2) * betas    
        loss_lis.append(np.abs(np.mean(G)))   
        betas_new = betas + step * G   
        betas = betas_new
    betas_final = betas
    return itera_lis, loss_lis, betas_final

def classifier2(X,Y,b): # for 6 and 8 only
    mmm = -np.dot(X,b)
    exp = np.exp(mmm)
    P_y = 1/(1 + exp)
    Y_hat_lis = []
    Y_hat = 0
    for p in P_y:
        if p >= 0.5:
            Y_hat = 8
            Y_hat_lis.append(Y_hat)
        if 0 <= p < 0.5:
            Y_hat = 6
            Y_hat_lis.append(Y_hat)
    return Y_hat_lis

 
def Accuracy2(Y_hat,Y): # for 6 and 8 only
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

def change_6_8_to_0_1(lis):
    new_lis = []
    for ind in range(len(lis)):
        if lis[ind] == 6:
            a = 0
            new_lis.append(a)
        else:
            a = 1
            new_lis.append(a)
    return new_lis
        

X_train_6_8s, Y_train_6_8s, total_train_6s, total_train_8s = pick_two_specific_digits(X_train,Y_train,6,8) # pick 6s and 8s only
X_test_6_8s, Y_test_6_8s, total_test_6s, total_test_8s = pick_two_specific_digits(X_test,Y_test,6,8) # pick 6s and 8s only

insert_1s_train2 = np.ones((X_train_6_8s.shape[0], 1))
X_train_6_8s = np.hstack((insert_1s_train2, X_train_6_8s))

insert_1s_test2 = np.ones((X_test_6_8s.shape[0], 1))
X_test_6_8s = np.hstack((insert_1s_test2, X_test_6_8s))

iterr2, los2, Beta2 = Gradient_Ascent2(X_train_6_8s,Y_train_6_8s,sigma2=0.001,step=0.001,N=60)


Y_hats_6_8 = classifier2(X_test_6_8s, Y_test_6_8s, Beta2)

Y_hats_6_8_for_01_loss = np.array(change_6_8_to_0_1(Y_hats_6_8)).reshape(Y_test_6_8s[0].shape)
Y_test_6_8s_for_01_loss = np.array(change_6_8_to_0_1(Y_test_6_8s[0])).reshape(Y_test_6_8s[0].shape)


aa, Expectations2 = Accuracy2(Y_hats_6_8, Y_test_6_8s)
wrong2 = len(Expectations2) -  Expectations2.count(0)

E2 = np.abs(sum(Y_hats_6_8_for_01_loss- Y_test_6_8s_for_01_loss)/len(Y_test_6_8s_for_01_loss))

print('Beta vector for 6 and 8 classification =\n%s' % Beta2)
print('Accuracy for 6 and 8 classification is %.6f' % aa)
print('Predicted Y (Y_tilde, 6 and 8s) = %s' % Y_hats_6_8)
print('Predicted Y (Y_tilde, 0 and 1s) = %s' % list(Y_hats_6_8_for_01_loss))
print('Y-Y_tilde = %s (0-1 loss)' % list(Y_hats_6_8_for_01_loss- Y_test_6_8s_for_01_loss))
print('E[|Y-Y_tilde|] = %f (0-1 loss)' % E2)
print('Wrongly classified amount: %d\n' % wrong2)

plt.figure(2)
plt.plot(iterr2,los2)
plt.xlabel('Epoch'),plt.ylabel('Loss')
plt.title('6 and 8 classification')
plt.legend(['Accuracy is %.6f' % aa], loc=1)



