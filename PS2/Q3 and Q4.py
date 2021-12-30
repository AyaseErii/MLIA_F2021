# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 23:09:20 2021

@author: jjerr
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 19:07:37 2021

@author: Jerry Yin
"""

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.interpolate import griddata

# Q3
def operatorK(img, a=16):
    height, width = img.shape

    # fft and shift
    img_fourier = np.fft.fft2(img)
    i_fftshift = np.fft.fftshift(img_fourier)

    # create mask
    mask = np.zeros(img.size, img.dtype).reshape(img.shape)
    for i in range(1, height):
        for j in range(1, width):
            # Creat squared mask
            if ((height - a) / 2) <= i <= ((height + a) /2) and ((width - a) / 2) <= j <= ((width + a) / 2):
                mask[i, j] = 1
            else:
                mask[i, j] = 0
    
    # reconstruct image
    re_image = np.fft.ifft2(np.fft.ifftshift(mask * i_fftshift))

    return np.abs(re_image)



def Euler_vt(velocity,h):
    t_lis = []
    vx_lis = []
    vy_lis = []
    phix_lis = []
    phiy_lis = []
    phi_lis = []
    v_3dim = np.squeeze(velocity)
    vy = v_3dim[:,:,0]
    vx = v_3dim[:,:,1]
    xx = np.linspace(0,99,100)
    yy = np.linspace(0,99,100)
    phiy, phix = np.meshgrid(xx,yy)
    phi = np.dstack((phix,phiy))
    t = 0
    for t in np.arange(0,1+h,h):
        if t == 0:
            t_lis.append(t)
            vx_lis.append(vx) 
            vy_lis.append(vy) 
            phix_lis.append(phix)
            phiy_lis.append(phiy)
            phi_lis.append(phi)
        else:                
            g_vx_x = vx - np.roll(vx,shift=-1,axis=0) # dvx/dx
            g_vx_y = vx - np.roll(vx,shift=-1,axis=1) # dvx/dy
            g_vy_x = vy - np.roll(vy,shift=-1,axis=0) # dvy/dx
            g_vy_y = vy - np.roll(vy,shift=-1,axis=1) # dvy/dy

            
            Dvt_vt_0 = g_vx_x * vx + g_vy_x * vy # 1st row of Dvt.T * vt
            Dvt_vt_1 = g_vx_y * vx + g_vy_y * vy # 2nd row of Dvt.T * vt
            
            
            div_0 = vx * vx - np.roll(vx * vx,shift=-1,axis=0) + vx * vy - np.roll(vx * vy,shift=-1,axis=1)  # 1st row of div(vt * vt.T)
            div_1 = vy * vx - np.roll(vy * vx,shift=-1,axis=0) + vy * vy - np.roll(vy * vy,shift=-1,axis=1)  # 2nd row of div(vt * vt.T)
            
            diff_total_0 = -operatorK(Dvt_vt_0 + div_0, a=16) # 1st row of  -K[Dvt.T * vt + div(vt * vt.T)]
            diff_total_1 = -operatorK(Dvt_vt_1 + div_1, a=16) # 2nd row of  -K[Dvt.T * vt + div(vt * vt.T)]
            
            
            diff_phix = griddata(np.reshape(phi,(10000,2)), vx.flatten(), (phix, phiy), method='linear') # dphix/dt, interpolation
            diff_phiy = griddata(np.reshape(phi,(10000,2)), vy.flatten(), (phix, phiy), method='linear') # dphiy/dt, interpolation
            
            
            t_lis.append(t)
            vx_new = vx + h * diff_total_0 # Eular method to get next vx
            vy_new = vy + h * diff_total_1 # Eular method to get next vy
            
            phix_new = phix + h * np.reshape(diff_phix,(100,100)) # Eular method to get next phix
            phiy_new = phiy + h * np.reshape(diff_phiy,(100,100)) # Eular method to get next phiy
            
            
            phi = np.dstack((phix_new,phiy_new)) # update phi
            phi_lis.append(phi)
            

            vx_lis.append(vx_new)
            vy_lis.append(vy_new)
            phix_lis.append(phix_new)
            phiy_lis.append(phiy_new)

            vx = vx_new # update vx
            vy = vy_new # update vy
            

            phix = phix_new # update phix
            phiy = phiy_new # update phiy
            
    return t_lis, phix_lis, phiy_lis, phi_lis

# Read v
'''Put v0Spatial.mhd and .raw and source.mhd and .raw into the same fold'''
#velocity = sitk.GetArrayFromImage(sitk.ReadImage('v0Spatial.mhd')) 
velocity = sitk.GetArrayFromImage(sitk.ReadImage('D:\\OneDrive - University of Virginia\\Courses\\ECE6501 Machine Learning in Image Analysis (Miaomiao Zhang)\\PS2\\PS2\\code+data_Q3\\data\\initialV\\v0Spatial.mhd')) 
# Read source img
'''Put v0Spatial.mhd and .raw and source.mhd .raw into the same fold'''
#source= sitk.GetArrayFromImage(sitk.ReadImage(source.mhd')) 
source= sitk.GetArrayFromImage(sitk.ReadImage('D:\\OneDrive - University of Virginia\\Courses\\ECE6501 Machine Learning in Image Analysis (Miaomiao Zhang)\\PS2\\PS2\\code+data_Q3\\data\\sourceImage\\source.mhd')) 

source = np.reshape(source,(100,100))
t, px, py, p = Euler_vt(velocity,h=0.02)

# Deform image
def_img = griddata(np.reshape(p[-1],(10000,2)), source.flatten(), (px[0], py[0]), method='linear')





# Q4
def Q4_Euler_vt(velocity,h):
    t_lis = []
    vx_lis = []
    vy_lis = []
    phix_lis = []
    phiy_lis = []
    phi_lis = []
    v_3dim = np.squeeze(velocity)
    vy = v_3dim[:,:,0]
    vx = v_3dim[:,:,1]
    xx = np.linspace(0,99,100)
    yy = np.linspace(0,99,100)
    phiy, phix = np.meshgrid(xx,yy)
    phi = np.dstack((phix,phiy))
    t = 0
    for t in np.arange(0,1.1,h):
        if t == 0:
            t_lis.append(t)
            vx_lis.append(vx)
            vy_lis.append(vy)            
            phix_lis.append(phix)
            phiy_lis.append(phiy)
            phi_lis.append(phi)
        else:     

            g_vx_x = vx - np.roll(vx,shift=-1,axis=0)
            g_vx_y = vx - np.roll(vx,shift=-1,axis=1)
            g_vy_x = vy - np.roll(vy,shift=-1,axis=0)
            g_vy_y = vy - np.roll(vy,shift=-1,axis=1)
            
            g_phix_x = phix - np.roll(phix,shift=-1,axis=0)
            g_phix_y = phix - np.roll(phix,shift=-1,axis=1)
            g_phiy_x = phiy - np.roll(phiy,shift=-1,axis=0)
            g_phiy_y = phiy - np.roll(phiy,shift=-1,axis=1)
            
            
            Dvt_vt_0 = g_vx_x * vx + g_vy_x * vy # 1st row of Dvt.T * vt
            Dvt_vt_1 = g_vx_y * vx + g_vy_y * vy # 2nd row of Dvt.T * vt
            
            
            div_0 = vx * vx - np.roll(vx * vx,shift=-1,axis=0) + vx * vy - np.roll(vx * vy,shift=-1,axis=1)  # 1st row of div(vt * vt.T)
            div_1 = vy * vx - np.roll(vy * vx,shift=-1,axis=0) + vy * vy - np.roll(vy * vy,shift=-1,axis=1)  # 2nd row of div(vt * vt.T)
            
            diff_total_0 = operatorK(Dvt_vt_0 + div_0, a=16) # 1st row of  -K[Dvt.T * vt + div(vt * vt.T)]
            diff_total_1 = operatorK(Dvt_vt_1 + div_1, a=16) # 2nd row of  -K[Dvt.T * vt + div(vt * vt.T)]
            
            diffphi_0 = - (g_phix_x * vx + g_phix_y * vy) # 1st row of -Dphi * vt
            diffphi_1 = - (g_phiy_x * vx + g_phiy_y * vy) # 2nd row of -Dphi * vt
            
            t_lis.append(t)
            vx_new = vx + h * diff_total_0  # Eular method to get next vx
            vy_new = vy + h * diff_total_1 # Eular method to get next vy
            phix_new = phix + h * diffphi_0 # Eular method to get next phix
            phiy_new = phiy + h * diffphi_1 # Eular method to get next phiy
            
            vx_lis.append(vx_new)
            vy_lis.append(vy_new)
            phix_lis.append(phix_new)
            phiy_lis.append(phiy_new)
            
            phi = np.dstack((phix_new,phiy_new))
            phi_lis.append(phi)

            vx = vx_new
            vy = vy_new
            phix = phix_new
            phiy = phiy_new
            
    return t_lis, phix_lis, phiy_lis, phi_lis


t4, px4, py4, p4 = Q4_Euler_vt(velocity,h=0.02)

# Recontruct image
recons = griddata(np.reshape(p4[0],(10000,2)), def_img.flatten(), (px4[-1], py4[-1]), method='linear')



# Plot 3 plots
plt.subplot(1,3,1)
plt.imshow(source, cmap='gray')
plt.title('Original'),plt.xlim(),plt.ylim()

plt.subplot(1,3,2)
plt.imshow(def_img, cmap='gray')
plt.title('Deformed image using \u03A6(t=%.1f)' % t[-1]),plt.xlim(),plt.ylim()

plt.subplot(1,3,3)
plt.imshow(recons, cmap='gray')
plt.title('Reconstructed image using \u03A6(t=%.1f)' % t[-1]),plt.xlim(),plt.ylim()


