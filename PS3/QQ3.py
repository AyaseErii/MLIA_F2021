# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 12:21:23 2021

@author: Jerry Yin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def rotation(sour_im,degree):
    ro_theta = degree * np.pi / 180    
    h, w = sour_im.shape
    
    w_new = int(w * np.cos(ro_theta) + h * np.sin(ro_theta)) + 1
    h_new = int(w * np.sin(ro_theta) + h * np.cos(ro_theta)) + 1
    
    rotated_im = np.zeros((h_new, w_new))

    # img coordinate to mathmatical coordinate
    s_to_d = np.array([[1,0,0],[0,-1,0],[-0.5 * w, 0.5 * h, 1]])
    s_to_d = s_to_d.dot(np.array([[np.cos(ro_theta),-np.sin(ro_theta),0],[np.sin(ro_theta),np.cos(ro_theta),0],[0,0,1]]))
    s_to_d = s_to_d.dot(np.array([[1,0,0],[0,-1,0],[0.5 * w_new, 0.5 * h_new, 1]]))
    
    # mathmatical coordinate to img coordinate
    d_to_s = np.array([[1,0,0],[0,-1,0],[-0.5 * w_new, 0.5 * h_new, 1]])
    d_to_s = d_to_s.dot(np.array([[np.cos(ro_theta),np.sin(ro_theta),0],[-np.sin(ro_theta),np.cos(ro_theta),0],[0,0,1]]))
    d_to_s = d_to_s.dot(np.array([[1,0,0],[0,-1,0],[0.5 * w, 0.5 * h, 1]]))
    
    for x in range(w):
        for y in range(h):
            pos_new = np.array([x,y,1]).dot(s_to_d)
            rotated_im[int(pos_new[1])][int(pos_new[0])] = sour_im[y][x] # interpolation
    
    for x in range(w_new):
        for y in range(h_new):
            pos_src = np.array([x,y,1]).dot(d_to_s)
            if pos_src[0] >= 0 and pos_src[0] < w and pos_src[1] >= 0 and pos_src[1] < h:
                rotated_im[y][x] = sour_im[int(pos_src[1])][int(pos_src[0])] # interpolation
    
    rotated_im = rotated_im[h_new-h : h_new+h, w_new-w : w_new+w] # make the image in shape of (28,28)
    return rotated_im


def translation(t_x,t_y,s):
    h, w = s.shape 
    new_img = np.zeros((h,w))    
    for x in range(w):
            for y in range(h):            
                if x <= w-1 and (x+t_x) <= w-1 and y <= h-1 and (y+t_y) <= h-1:
                    new_img[np.abs(int(y+t_y))][np.abs(int(x+t_x))] = s[y][x]   # interpolation
    return new_img


def Aug_Training_Set(X,Y,NN):
    X_aug = X
    Y_aug = Y  
    for n in range(NN):
        i = np.random.randint(len(X))
        t_x = np.random.uniform(-5,5,1) # T in [-5,5]
        t_y = np.random.uniform(-5,5,1) # T in [-5,5]   
        degree = np.random.uniform(0,60,1) # theta in [0,60]
        s = np.reshape(np.array(X.T[i]),(28,28))  
        r1 = rotation(s,degree[0])
        r2 = translation(t_x[0],t_y[0],r1) # rotation + translation img
        
        r2_df = pd.DataFrame(r2.reshape(1,784))
        Y_train_df = pd.DataFrame(Y[0][i].reshape(1,1))
        
        X_aug_new = X_aug.append(r2_df)
        X_aug_new = X_aug_new.reset_index(drop=True)
        
        Y_aug_new = Y_aug.append(Y_train_df)
        Y_aug_new = Y_aug_new.reset_index(drop=True)
        
        X_aug = X_aug_new  # Augmented X training data set (updated X)
        Y_aug = Y_aug_new  # Augmented Y training data set (updated Y)        
    return X_aug, Y_aug


def sigmoid(z):  
    return np.exp(-z) / (1.0 + np.exp(-z))


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

def change_6_8_to_0_1(lis): # For computing E[|Y-Y_tilde|]
    new_lis = []
    for ind in range(len(lis)):
        if lis[ind] == 6:
            a = 0
            new_lis.append(a)
        if lis[ind] == 8:
            a = 1
            new_lis.append(a)
    return new_lis



# Read files. If it cannot be run, please change the path to absolute path, thanks!
X_train = pd.read_csv('trainX.csv', header=None)
Y_train = pd.read_csv('trainY.csv', header=None)

X_test = pd.read_csv('testX.csv', header=None)
Y_test = pd.read_csv('testY.csv', header=None)


X_train_6_8s, Y_train_6_8s, total_train_6s, total_train_8s = pick_two_specific_digits(X_train,Y_train,6,8) # pick 6s and 8s only
X_test_6_8s, Y_test_6_8s, total_test_6s, total_test_8s = pick_two_specific_digits(X_test,Y_test,6,8) # pick 6s and 8s only


# Insert 1s col at the beginning of X test data set
insert_1s_test2 = np.ones((X_test_6_8s.shape[0], 1))
X_test_6_8s = np.hstack((insert_1s_test2, X_test_6_8s))


#Start image augmentation
NN = [0,100,300,600]
iterr_l = []
los_l = []
Beta_l = []
X_train_6_8s_aug_lis = [] # without 1s col
Y_train_6_8s_aug_lis = []
aa_l = []
np.random.seed(3) 


for n in NN:
    X_train_6_8s_aug, Y_train_6_8s_aug = Aug_Training_Set(X_train_6_8s, Y_train_6_8s,n)
    X_train_6_8s_aug_lis.append(X_train_6_8s_aug)
    Y_train_6_8s_aug_lis.append(Y_train_6_8s_aug)
    # Insert 1s col at the beginning of augmented X training data set
    insert_1s_train2 = np.ones((X_train_6_8s_aug.shape[0], 1))
    X_train_6_8s_aug = np.hstack((insert_1s_train2, X_train_6_8s_aug))       
    iterr, los, Beta = Gradient_Ascent2(X_train_6_8s_aug,Y_train_6_8s_aug, sigma2=0.001, step=0.001, N=100)
    iterr_l.append(iterr)
    los_l.append(los)
    Beta_l.append(Beta)    
    Y_hats_6_8 = classifier2(X_test_6_8s, Y_test_6_8s, Beta)    
    Y_hats_6_8_for_01_loss = np.array(change_6_8_to_0_1(Y_hats_6_8)).reshape(Y_test_6_8s[0].shape)
    Y_test_6_8s_for_01_loss = np.array(change_6_8_to_0_1(Y_test_6_8s[0])).reshape(Y_test_6_8s[0].shape)   
    aa, Expecations = Accuracy2(Y_hats_6_8, Y_test_6_8s)
    wrong = len(Expecations) - Expecations.count(0)
    aa_l.append(aa)
    E2 = np.abs(sum(Y_hats_6_8_for_01_loss - Y_test_6_8s_for_01_loss)/len(Y_test_6_8s_for_01_loss))    
    print('When N=%d, the accuracy for 6 and 8 classification is %.6f' % (n,aa))
    print('Predicted Y (Y_tilde, 6 and 8s) = %s' % Y_hats_6_8)
    print('Predicted Y (Y_tilde, 0 and 1s) = %s' % list(Y_hats_6_8_for_01_loss))
    print('E[|Y-Y_tilde|] = %s' % E2)
    print('Wrongly classified amount: %d\n' % wrong)

plt.figure(1)
plt.plot(NN, aa_l)
plt.title('Augmented N vs. Accuracy')
plt.xlabel('Augmented N'),plt.ylabel('Accuracy')        

    
    
    





                      







