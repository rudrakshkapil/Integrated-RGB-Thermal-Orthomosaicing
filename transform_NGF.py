# %% [markdown]
# # This assignment is on image registration using perpective (also known as homography) transformation
# 
# #### This notebook shows the use of differentiable mutual information for image registration. Your main task is to change that registration cost function to normalizing gradient. The code will run a lot faster. 
# 
# #### You are required to edit some portions of a two functions as indicated in the instructions. You will also answer two questions provided in two cells.
# 
# #### You will find outputs and expected behaviour of a sucessful implementation towards the end of the notebook.
# 
# #### You will edit only the portions of the code/cell you are instructed to do. Top of each cell indicates wherether you should or should not edit that cell.
# 
# #### Look for submission instructions in eClass.

# %%
# Do not edit this cell
import os
from tqdm import tqdm
import random
from skimage import io
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage.transform import pyramid_gaussian
import cv2

# for local hist eq
from skimage.morphology import disk
from skimage.filters import rank
from skimage.util import img_as_ubyte
from skimage.exposure import equalize_hist


def histogram_mutual_information(image1, image2):
    '''
    Function to calculate histogram-based mutual information between two images
    '''
    img1 = image1.ravel()
    img2 = image2.ravel()
    hgram, x_edges, y_edges = np.histogram2d(img1, img2, bins=100)
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))



class HomographyNet(nn.Module):
    '''
    Homography Network
    '''
    def __init__(self, device, dof):
        super(HomographyNet, self).__init__()
        self.device = device

        # # perspective transform basis matrices
        self.B = torch.zeros(8,3,3).to(device)
        # self.B[0,0,2] = 1.0
        # self.B[1,1,2] = 1.0
        # self.B[2,0,1], self.B[2,1,0] = -1.0, 1.0

        # if dof == 'similarity':
        #     self.B[3,2,2] = -1.0

        # # TODO: 
        # elif dof == 'affine':
        #     self.B[3,0,0], self.B[3,1,1] = 1.0, 1.0
        #     self.B[4,0,0], self.B[4,1,1] = 1.0, -1.0
        #     self.B[5,0,1], self.B[2,1,0] = 1.0, 1.0

        # elif dof == 'perspective':
        #     self.B[3,0,0], self.B[3,1,1], self.B[3,2,2] = 1.0, 1.0, -2.0
        #     self.B[4,0,0], self.B[4,1,1] = 1.0, -1.0
        #     self.B[5,0,1], self.B[2,1,0] = 1.0, 1.0
        #     self.B[6,2,0] = 1.0
        #     self.B[7,2,1] = 1.0

        self.B[0,0,2] = 1.0
        self.B[1,1,2] = 1.0
        self.B[2,0,1] = 1.0
        self.B[3,1,0] = 1.0
        self.B[4,0,0], self.B[4,1,1] = 1.0, -1.0
        self.B[5,1,1], self.B[5,2,2] = -1.0, 1.0

        if dof == 'perspective':
            self.B[6,2,0] = 1.0
            self.B[7,2,1] = 1.0

        self.v = torch.nn.Parameter(torch.zeros(8,1,1).to(device), requires_grad=True)

    # This function computes forward transform matrix
    def forward(self):
        return MatrixExp(self.B,self.v, self.device)

    # This function computes inverse transform matrix
    def inverse(self):
        return MatrixExp(self.B,-self.v, self.device)


def MatrixExp(B,v,device):
    '''
    Matrix Exponential Function
    '''
    C = torch.sum(B*v,0)
    A = torch.eye(3).to(device)
    H = torch.eye(3).to(device)
    for i in torch.arange(1,10):
        A = torch.mm(A/i,C)
        H = H + A
    return H    

def PerspectiveWarping(I, H, xv, yv):
    '''
    FUnction to apply transformation in the homogeneous coordinates (batch-wise)
    '''
    xvt = (xv*H[0,0]+yv*H[0,1]+H[0,2])/(xv*H[2,0]+yv*H[2,1]+H[2,2])
    yvt = (xv*H[1,0]+yv*H[1,1]+H[1,2])/(xv*H[2,0]+yv*H[2,1]+H[2,2])
    J = F.grid_sample(I,torch.stack([xvt,yvt],2).unsqueeze(0),align_corners=False).squeeze()
    return J

def multi_resolution_loss(I_lst, J_lst, xy_lst, homography_net, L, two_way_loss):
    '''
    Function to compute NGF multi-resolution loss (batch-wise)
    '''
    loss = 0.0
    for s in np.arange(L-1,-1,-1):
        Jw_ = PerspectiveWarping(J_lst[s].unsqueeze(0), homography_net(), xy_lst[s][:,:,0], xy_lst[s][:,:,1]).squeeze()
        ng = NGF(I_lst[s],Jw_)
        loss += ng

        # inverse direction loss as well
        if two_way_loss:
            Iw_ = PerspectiveWarping(I_lst[s].unsqueeze(0), homography_net.inverse(), xy_lst[s][:,:,0], xy_lst[s][:,:,1]).squeeze()
            ng2 = NGF(J_lst[s],Iw_)
            loss += ng2
    return loss

def gradient(I):
    '''
    Function to compute gradient of images (batch-wise)
    using central finite difference formula
    '''
    h = I.shape[1]
    w = I.shape[2]
        
    I = F.pad(I.unsqueeze(0),(1,1,1,1),'replicate').squeeze()
    
    # central finite difference formula for Ix and Iy
    Ix = 0.5*(I[:,1:h+1,2:w+2]-I[:,1:h+1,0:w])
    Iy = 0.5*(I[:,2:h+2,1:w+1]-I[:,0:h,1:w+1])

    return Ix,Iy


def NGF(I,J):
    '''
    Normalized Gradient Field (NGF) Function (batch-wise)
    '''
    # compute gradients
    Ix,Iy = gradient(I)
    Jx,Jy = gradient(J)
    
    # square then sum
    Imag = torch.sqrt(Ix**2 + Iy**2 + 1e-8)
    Jmag = torch.sqrt(Jx**2 + Jy**2 + 1e-8)
    
    # compute and return loss
    ngf_loss = torch.mean((Ix/Imag - Jx/Jmag)**2) + torch.mean((Iy/Imag - Jy/Jmag)**2)
    return ngf_loss


def hist_match1(source, template):
    hist_source, _ = np.histogram(source, bins = 256, range=(0, 256))
    hist_source = hist_source.astype(np.float64)
    #hist_source/= sum(hist_source)
    hist_template, _ =  np.histogram(template, bins = 256, range=(0, 256))
    hist_template = hist_template.astype(np.float64)
    #hist_template/= sum(hist_template)
    c_template = np.cumsum(hist_template)
    c_template /= c_template[-1]
    c_source = np.cumsum(hist_source)
    c_source /= c_source[-1]
    A = np.zeros((256,), dtype = int)
    for a in range(256):
        a_ = 0
        while c_source[a] > c_template[a_]:
            a_ +=1
        A[a] = a_
    
    J = np.zeros_like(source)
    for i in range(source.shape[0]):
        for j in range(source.shape[1]):
            a = source[i,j]
            J[i,j] = A[a]

    return J


def map_all_images_using_NGF(I_dir, J_dir, save_dir, thermal_texture_dir, date, params={}, thermal_ext='tiff', dof='affine', eval=False):
    # GPU # TODO: revert
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # extract required params
    batch_size = params.get('BATCH_SIZE', 64)
    hist_eq_opt = params.get('HIST_EQ', True)
    L = params.get('LEVELS', 11)
    downscale = params.get('DOWNSCALE', 1.5)
    learning_rate = params.get('LEARNING_RATE', 0.5e-2)
    n_iters = params.get('N_ITERS', 400)
    lower_bound = params.get('MI_LOWER_BOUND', 0.2)
    search_all_images = params.get('SEARCH_ALL_IMAGES', True)
    verbose = params.get('VERBOSE', True)
    hom_save_path = f'{date}/logs/log_homography_ngf.txt'
    two_way_loss = params.get('TWO_WAY_LOSS', True)

 
    # load image names
    I_files = [f"{I_dir}/{fname}" for fname in os.listdir(I_dir) if fname.endswith(".tif")]
    J_files = [f"{J_dir}/{fname.split('.')[0]}.{thermal_ext}" for fname in os.listdir(I_dir) if fname.endswith(".tif")]
    assert len(I_files) == len(J_files)

    # splice first batchsize amount
    if not search_all_images:
        I_files = I_files[:batch_size]
        J_files = J_files[:batch_size]

    l_ = len(cv2.imread(J_files[0]).shape)
    

    # hist equalization for both sets of images
    print("Loading original images (two sets)...")
    if hist_eq_opt:
        I_imgs = np.array([equalize_hist(img_as_ubyte(rgb2gray(io.imread(fname)))).astype(np.float32) for fname in tqdm(I_files)])
        # if l_ == 2:
        J_imgs = np.array([equalize_hist(io.imread(fname)).astype(np.float32) for fname in tqdm(J_files)])
        # else:
        #     J_imgs = np.array([equalize_hist(img_as_ubyte(rgb2gray(io.imread(fname)))).astype(np.float32) for fname in tqdm(J_files)])
    # normal -- load all images
    else:
        I_imgs = np.array([rgb2gray(io.imread(fname)).astype(np.float32) for fname in tqdm(I_files)])
        # if l_ == 2:
        J_imgs = np.array([io.imread(fname).astype(np.float32)/(256*256-1.0) for fname in tqdm(J_files)]) # TODO: remove 10
        # else:
        #     I_imgs = np.array([rgb2gray(io.imread(fname)).astype(np.float32) for fname in tqdm(J_files)])
        # I_imgs = np.array([rgb2gray(io.imread(fname)).astype(np.float32) for fname in tqdm(I_files)])
        # J_imgs = np.array([io.imread(fname).astype(np.float32)/(256*256-1.0) for fname in tqdm(J_files)])

        # J_imgs = np.array([hist_match1((J_imgs[i]*256).astype(np.uint16), (I_imgs[i]*256).astype(np.uint16)) for i in range(len(I_imgs))])
        # J_imgs = (J_imgs/256.0).astype(np.float32)
        # print(J_imgs.shape)



    # compute mutual infomration for all original pairs 
    if search_all_images:
        upper_bound = 1.0
        zipped_list = []
        print("Calculating MI of each original pair and sorting...")
        for k in tqdm(range(I_imgs.shape[0])):
            MI_value = histogram_mutual_information(I_imgs[k], J_imgs[k])
            if lower_bound <= MI_value <= upper_bound:
                zipped_list.append((k, MI_value))

        # sort indices by MI (only if > lower_bound and < upper bound)
        zipped_list.sort(key=lambda a: a[1], reverse=True)

        # reduce batch size if needed
        batch_size = min(batch_size, len(zipped_list))
        batch_idxs = [item[0] for item in zipped_list[:batch_size]]

        # sample top MI images to use for computing homograpy
        I_imgs = I_imgs[batch_idxs]
        J_imgs = J_imgs[batch_idxs]
    assert np.amax(I_imgs) <= 1.0 and np.amax(J_imgs) <= 1.0 # ensure data type correct


    # create pyramid for each image as list of [(BS, H, W), (BS, H/ds, W/ds), ...]
    print("Creating image pyramid for each image...")
    pyramid_I_list, pyramid_J_list = None, None
    for k in tqdm(range(batch_size)):
        pyramid_I_temp = list(pyramid_gaussian(I_imgs[k], downscale=downscale, multichannel=False))
        pyramid_J_temp = list(pyramid_gaussian(J_imgs[k], downscale=downscale, multichannel=False))
        pyramid_I_temp = [np.expand_dims(pyramid_I_temp[s], axis=0) for s in range(len(pyramid_I_temp))]
        pyramid_J_temp = [np.expand_dims(pyramid_J_temp[s], axis=0) for s in range(len(pyramid_J_temp))]

        if pyramid_J_list is None:
            pyramid_I_list = pyramid_I_temp.copy()
            pyramid_J_list = pyramid_J_temp.copy()
        else:
            pyramid_I_list = tuple([np.concatenate((pyramid_I_list[s], pyramid_I_temp[s])) for s in range(L)])
            pyramid_J_list = tuple([np.concatenate((pyramid_J_list[s], pyramid_J_temp[s])) for s in range(L)])
        

    # create a list of necessary objects you will need and commit to GPU
    I_lst,J_lst,h_lst,w_lst,xy_lst,ind_lst=[],[],[],[],[],[]
    for s in range(L):
        I_, J_ = torch.tensor(pyramid_I_list[s].astype(np.float32)).to(device), torch.tensor(pyramid_J_list[s].astype(np.float32)).to(device)
        I_lst.append(I_)
        J_lst.append(J_)

        h_, w_ = I_lst[s].shape[1], I_lst[s].shape[2]
        h_lst.append(h_)
        w_lst.append(w_)

        y_, x_ = torch.meshgrid([torch.arange(0,h_).float().to(device), torch.arange(0,w_).float().to(device)])
        y_, x_ = 2.0*y_/(h_-1) - 1.0, 2.0*x_/(w_-1) - 1.0
        xy_ = torch.stack([x_,y_],2)
        xy_lst.append(xy_)

    # create homography net and optimizer (Adam)
    homography_net = HomographyNet(device, dof).to(device)
    optimizer = optim.Adam(homography_net.parameters(), learning_rate, amsgrad=True)

    # training code
    losses = []
    print("Training homography network...")
    for itr in tqdm(range(n_iters)):
        optimizer.zero_grad()
        loss = multi_resolution_loss(I_lst, J_lst, xy_lst, homography_net, L, two_way_loss)
        losses.append(loss.detach().cpu().numpy())
        if itr%10 == 0 and verbose:
            print("Itr:",itr,"Loss value:","{:.4f}".format(loss.item()))
        loss.backward()
        optimizer.step()
    print("Itr:",itr+1,"Loss value:","{:.4f}".format(loss.item()))
    losses = np.array(losses)
    plt.plot(losses)
    plt.title("Training loss for NGF")
    plt.savefig(f"{date}/logs/loss_NGF.png")

    # save homography net
    torch.save(homography_net.state_dict(), f'{date}/combined/ngf_homography_net.pth')

    from PIL import Image
    with torch.no_grad():
        # get all images to warp (batch)
        print("Reading all original thermal images to warp...")
        J_imgs = np.array([cv2.imread(f"{thermal_texture_dir}/{fname}", cv2.IMREAD_UNCHANGED) for fname in tqdm(os.listdir(thermal_texture_dir))], dtype=np.float32)

        idx = 0
        print("Saving warped images...")
        for i in tqdm(range(0, len(J_imgs), batch_size)):
            if i+batch_size < len(J_imgs):
                J_t = torch.tensor(J_imgs[i:i+batch_size]).to(device)
            else:
                J_t = torch.tensor(J_imgs[i:]).to(device)

            # get homography matrix and warp entire batch
            # homography_net = HomographyNet(device).to(device)
            # homography_net.load_state_dict(f"{date}/homography_net.pth")
            H = homography_net()
            J_w = PerspectiveWarping(J_t.unsqueeze(0), H , xy_lst[0][:,:,0], xy_lst[0][:,:,1]).squeeze()
            J_w = J_w.detach().cpu().numpy().astype(np.uint16)

            # save warped images
            try: os.makedirs(save_dir) 
            except: pass
            save_file_names = os.listdir(thermal_texture_dir)
            for j in tqdm(range(len(J_w)), leave=False):
                # im = Image.fromarray((J_w[j]*65535).astype(np.uint16))
                # im.save(f"{save_dir}/{save_file_names[idx]}.tif", compression='tiff_lzw')

                cv2.imwrite(f"{save_dir}/{save_file_names[idx]}", J_w[j])
                idx += 1

        # write homography to file
        with open(hom_save_path, 'w') as f:
            H = H.detach().cpu().numpy()
            f.write(str(H))

    if eval:
        # eval
        # I = rgb2gray(io.imread(f"{I_dir}/img_00007.jpeg.tif"))
        # J = io.imread(f"{J_dir}/img_00007.tiff")
        I = I_imgs[0]
        J = J_imgs[0]
        # J = J.astype(np.float(32))/(256*256-1.0)


        I_t = torch.tensor(I).to(device)
        J_t = torch.tensor(J).to(device)

        J_w = PerspectiveWarping(J_t.unsqueeze(0).unsqueeze(0), H , xy_lst[0][:,:,0], xy_lst[0][:,:,1]).squeeze()

        D = J_t - I_t
        D_w = J_w - I_t

        Ra = I_t.clone()
        Rb = I_t.clone()
        b = 150
        for i in torch.arange(0,I_t.shape[0]/b,1).int():
          for j in torch.arange(i%2,np.floor(I_t.shape[1]/b),2).int():
            Rb[i*b:(i+1)*b,j*b:(j+1)*b] = J_t[i*b:(i+1)*b,j*b:(j+1)*b].clone()
            Ra[i*b:(i+1)*b,j*b:(j+1)*b] = J_w[i*b:(i+1)*b,j*b:(j+1)*b].clone()

        fig=plt.figure(figsize=(10,10))
        fig.add_subplot(1,2,1)
        plt.imshow(Rb.cpu().data,cmap="gray")
        plt.title("Images before registration")
        fig.add_subplot(1,2,2)
        plt.imshow(Ra.cpu().data,cmap="gray")         
        plt.title("Images after registration")
        plt.show()

        fig=plt.figure(figsize=(10,10))
        fig.add_subplot(1,2,1)
        plt.imshow(I_t.cpu().data,cmap="gray")
        plt.title("Thermal Image before registration")
        fig.add_subplot(1,2,2)
        plt.imshow(J_w.cpu().data,cmap="gray")         
        plt.title("Target image")
        plt.show()




