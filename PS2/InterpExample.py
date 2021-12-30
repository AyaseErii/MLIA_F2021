
import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as nnf

def torchinterp(size, src, flow):
    vectors = [torch.arange(0, s) for s in size]
    grids = torch.meshgrid(vectors)
    grid = torch.cat((grids[2], grids[1], grids[0]), dim=0)
    grid = torch.unsqueeze(grid, 0)
    grid = torch.unsqueeze(grid, 2)
    grid = grid.type(torch.FloatTensor)
    mode='bilinear'
    # new locations
    ''' Deformation = Identity transformation + displacement field'''
    new_locs =  grid + flow
    shape = flow.shape[2:]
    # normalize deformation grid values to [-1, 1] 
    for i in range(len(shape)):
        if (i == 0) or (i == 1):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[len(shape)-i-1] - 1) - 0.5)
        if (i == 2):
            new_locs[:, 2,:,:,:] = 0
    # move channels dim to last position
    if len(shape) == 2:
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]
    elif len(shape) == 3:
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
    return nnf.grid_sample(src, new_locs, mode=mode, align_corners=True)



######!!!! Loading a displacemnet field instead of phi directly!!! displacement field = Phi_1 - Phi_0 ###############
itkimage = sitk.ReadImage('./displace.mhd')
disp = torch.tensor(sitk.GetArrayFromImage(itkimage))
#%%
disp  = disp.permute(0, 3, 1, 2)
#%%
displace = disp.reshape(1,3,1,100,100)

######Loading a source image ###############
itkimage = sitk.ReadImage('./Image_3.mhd')
src= sitk.GetArrayFromImage(itkimage)
src= torch.tensor(src)
#%%
src = src.reshape(1,1,1,100,100)
size = [1,100,100]
deformed = torchinterp(size,src,displace)
#%%
print(src.shape)
######### Showing source ################
im_s = src.reshape(100,100).detach().numpy()
plt.figure()
plt.imshow(im_s)
plt.title("Source Image")
plt.colorbar()

######### Showing deformed ################
deformedim = deformed.reshape(100,100).detach().numpy()
plt.figure()
plt.imshow(deformedim)
plt.title("Deformed Image")
plt.colorbar()
plt.show()