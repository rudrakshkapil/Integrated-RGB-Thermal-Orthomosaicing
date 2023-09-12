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

def multi_resolution_loss(I_lst, J_lst, xy_lst, homography_net, L, two_way_loss, ngf_flag=True):
    '''
    Function to compute NGF multi-resolution loss (batch-wise)
    '''
    loss = 0.0
    for s in np.arange(L-1,-1,-1):
        Jw_ = PerspectiveWarping(J_lst[s].unsqueeze(0), homography_net(), xy_lst[s][:,:,0], xy_lst[s][:,:,1]).squeeze()
        if len(Jw_.shape) == 2: Jw_ = Jw_.unsqueeze(0)
        ng = NGF(I_lst[s],Jw_) if ngf_flag else ECC(I_lst[s],Jw_)
        loss += ng

        # inverse direction loss as well
        if two_way_loss:
            Iw_ = PerspectiveWarping(I_lst[s].unsqueeze(0), homography_net.inverse(), xy_lst[s][:,:,0], xy_lst[s][:,:,1]).squeeze()
            if len(Iw_.shape) == 2: Iw_ = Iw_.unsqueeze(0)
            ng2 = NGF(J_lst[s],Iw_) if ngf_flag else ECC(I_lst[s],Jw_)
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
    if len(I.shape)==2: # for bs = 1
        I = I.unsqueeze(0)
    
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
    ngf_loss = torch.mean((Ix/Imag - Jx/Jmag)**2)
    ngf_loss += torch.mean((Iy/Imag - Jy/Jmag)**2)
    return ngf_loss

def ECC(I,J):
    '''
    Function to compute ECC loss between two images (batch implementation)
    '''
    bsz, h, w = I.shape

    # get zero mean vectors
    Izm = torch.reshape(I, (bsz, h*w)) - torch.mean(I, dim=(1,2)).unsqueeze(1)
    Jzm = torch.reshape(J, (bsz, h*w)) - torch.mean(J, dim=(1,2)).unsqueeze(1)

    # get l2 nroms
    Izm_norm = torch.linalg.norm(Izm, dim=1, ord=2).unsqueeze(1)
    Jzm_norm = torch.linalg.norm(Jzm, dim=1, ord=2).unsqueeze(1)

    # Compute ECC from formula, 
    ecc_loss = Izm/Izm_norm - Jzm/Jzm_norm 
    ecc_loss = torch.linalg.norm(ecc_loss, dim=1, ord=2)**2
    ecc_loss = ecc_loss**2

    # mean loss over all batches and return
    ecc_loss = torch.mean(ecc_loss)
    return ecc_loss


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
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # extract required params
    batch_size = params.get('BATCH_SIZE', 64) 
    hist_eq_opt = params.get('HIST_EQ', False)
    L = params.get('LEVELS', 11)
    downscale = params.get('DOWNSCALE', 1.5)
    learning_rate = params.get('LEARNING_RATE', 0.5e-2)
    n_iters = params.get('N_ITERS', 200)
    lower_bound = params.get('MI_LOWER_BOUND', 0.0)
    search_all_images = params.get('SEARCH_ALL_IMAGES', False) 
    systematic_sample = params.get('SYSTEMATIC_SAMPLE', True)
    verbose = params.get('VERBOSE', True)
    hom_save_path = f'{date}/logs/log_homography_ngf.txt'
    two_way_loss = params.get('TWO_WAY_LOSS', True)
    ngf_flag = params.get('NGF_FLAG', True)

 
    # load image names
    I_files = [f"{I_dir}/{fname}" for fname in os.listdir(I_dir) if fname.endswith(".tif")]
    J_files = [f"{J_dir}/{fname.split('.')[0]}.{thermal_ext}" for fname in os.listdir(I_dir) if fname.endswith(".tif")]
    assert len(I_files) == len(J_files)

    # splice first batchsize amount
    if not search_all_images:
        # sequential or random - get appropriate indices
        if systematic_sample:
            rand_idxs= list(range(0, len(J_files), len(J_files)//batch_size))
        else:
            rand_idxs = list(np.random.choice(range(len(I_files)), batch_size, replace=False))

        # sample
        I_files = list(np.array(I_files)[rand_idxs])
        J_files = list(np.array(J_files)[rand_idxs])

        # make sure batch size as correct
        if len(I_files) != batch_size:
            I_files = I_files[:batch_size]
            J_files = J_files[:batch_size]

    

    # hist equalization for both sets of images
    print("Loading original images (two sets)...")
    if hist_eq_opt: # histogram equalization
        I_imgs = np.array([equalize_hist(rgb2gray(io.imread(fname))).astype(np.float32) for fname in tqdm(I_files)])
        J_imgs = np.array([equalize_hist(io.imread(fname)).astype(np.float32) for fname in tqdm(J_files)])

    else: # min-max normalization
        I_imgs = np.array([rgb2gray(io.imread(fname)).astype(np.float32) for fname in tqdm(I_files)])
        J_imgs = np.array([io.imread(fname).astype(np.float32) for fname in tqdm(J_files)]) 
        
        I_imgs[:] = (I_imgs[:] - np.amin(I_imgs, axis=(1,2)).reshape(-1,1,1)) / (np.amax(I_imgs, axis=(1,2)) - np.amin(I_imgs, axis=(1,2))).reshape(-1,1,1)
        J_imgs[:] = (J_imgs[:] - np.amin(J_imgs, axis=(1,2)).reshape(-1,1,1)) / (np.amax(J_imgs, axis=(1,2)) - np.amin(J_imgs, axis=(1,2))).reshape(-1,1,1)



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
        zipped_list.sort(key=lambda a: a[1], reverse=True) # TODO: Reverse true

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
        loss = multi_resolution_loss(I_lst, J_lst, xy_lst, homography_net, L, two_way_loss, ngf_flag)
        losses.append(loss.detach().cpu().numpy())
        if itr%10 == 0 and verbose:
            print("Itr:",itr,"Loss value:","{:.4f}".format(loss.item()))
        loss.backward()
        optimizer.step()
    print("Itr:",itr+1,"Loss value:","{:.4f}".format(loss.item()))
    losses = np.array(losses)
    plt.plot(losses)
    plt.title("Training loss for NGF")

    # save homography net
    torch.save(homography_net.state_dict(), f'{date}/ngf_homography_net.pth')
    homography_net = HomographyNet(device, dof).to(device)

    from PIL import Image
    with torch.no_grad():
        # get all images to warp (batch)
        print("Reading all original thermal images to warp...")
        J_imgs = np.array([cv2.imread(f"{thermal_texture_dir}/{fname}", cv2.IMREAD_UNCHANGED) for fname in tqdm(os.listdir(thermal_texture_dir))])
        dtype = J_imgs.dtype
        J_imgs = J_imgs.astype(np.float32)
        dtype2 = J_imgs.dtype

        idx = 0
        # save warped images
        try: os.makedirs(save_dir) 
        except: pass
        save_file_names = os.listdir(thermal_texture_dir)


        batch_size = 64
        if batch_size % len(save_file_names) == 1:
            batch_size += 1
        print("Saving warped images...")
        for i in tqdm(range(0, len(J_imgs), batch_size)):
            if i+batch_size < len(J_imgs):
                J_t = torch.tensor(J_imgs[i:i+batch_size]).to(device)
            else:
                J_t = torch.tensor(J_imgs[i:]).to(device)

            # get homography matrix and warp entire batch
            homography_net = HomographyNet(device, dof).to(device)
            # homography_net.load_state_dict(torch.load(f"2022_08_30/ngf_homography_net.pth")) # TODO: change to {date/}
            homography_net.load_state_dict(torch.load(f"{date}/ngf_homography_net.pth")) # TODO: change to {date/}
            H = homography_net()
            J_w = PerspectiveWarping(J_t.unsqueeze(0), H , xy_lst[0][:,:,0], xy_lst[0][:,:,1]).squeeze()
            if dtype != np.float32:
                J_w = J_w.detach().cpu().numpy().astype(np.uint16)
            else:
                J_w = J_w.detach().cpu().numpy().astype(np.float32)


            for j in tqdm(range(len(J_w)), leave=False):
                cv2.imwrite(f"{save_dir}/{save_file_names[idx]}", J_w[j])
                idx += 1



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





def run():

    exp_name = 'images_trial'
    for project in ['2022_07_20', '2022_07_26', '2022_08_09', '2022_08_17', '2022_08_30']:
        undist_rgb_dir = f"{project}/combined/opensfm/undistorted/images_rgb"
        thermal_final_dir = f"{project}/thermal/images_resized"
        thermal_texture_dir = f"{project}/thermal/images_resized"
        save_dir = f"{project}/combined/opensfm/undistorted/{exp_name}"
        params = {'NGF_FLAG':True}
        dof = 'affine'
        map_all_images_using_NGF(undist_rgb_dir, thermal_final_dir, save_dir, thermal_texture_dir, project, params=params, thermal_ext='tiff', dof=dof)
    return

    #  single res
    # print("\n\n\nA")
    # exp_name = 'images_ngf'
    # for project in ['2022_07_20', '2022_07_26', '2022_08_09', '2022_08_17', '2022_08_30']:
    #     undist_rgb_dir = f"{project}/combined/opensfm/undistorted/images_rgb"
    #     thermal_final_dir = f"{project}/thermal/images_resized"
    #     thermal_texture_dir = f"{project}/thermal/images_resized"
    #     save_dir = f"{project}/combined/opensfm/undistorted/{exp_name}"
    #     params = {'NGF_FLAG':True}
    #     dof = 'affine'
    #     map_all_images_using_NGF(undist_rgb_dir, thermal_final_dir, save_dir, thermal_texture_dir, project, params=params, thermal_ext='tiff', dof=dof)

    # print("\n\n\nAA")
    # exp_name = 'images_ecc'
    # for project in ['2022_07_20', '2022_07_26', '2022_08_09', '2022_08_17', '2022_08_30']:
    #     undist_rgb_dir = f"{project}/combined/opensfm/undistorted/images_rgb"
    #     thermal_final_dir = f"{project}/thermal/images_resized"
    #     thermal_texture_dir = f"{project}/thermal/images_resized"
    #     save_dir = f"{project}/combined/opensfm/undistorted/{exp_name}"
    #     params = {'NGF_FLAG':False}
    #     dof = 'affine'
    #     map_all_images_using_NGF(undist_rgb_dir, thermal_final_dir, save_dir, thermal_texture_dir, project, params=params, thermal_ext='tiff', dof=dof)




    # #  single res
    # print("\n\n\nAAAAA")
    # exp_name = 'images_single_res'
    # for project in ['2022_07_20', '2022_07_26', '2022_08_09', '2022_08_17', '2022_08_30']:
    #     undist_rgb_dir = f"{project}/combined/opensfm/undistorted/images_rgb"
    #     thermal_final_dir = f"{project}/thermal/images_resized"
    #     thermal_texture_dir = f"{project}/thermal/images_resized"
    #     save_dir = f"{project}/combined/opensfm/undistorted/{exp_name}"
    #     params = {'LEVELS':1}
    #     dof = 'affine'
    #     map_all_images_using_NGF(undist_rgb_dir, thermal_final_dir, save_dir, thermal_texture_dir, project, params=params, thermal_ext='tiff', dof=dof)

    # batch size
    print("\n\n\nBBBBB")
    for bsz in [32]:#[1, 4, 16]:
        exp_name = f'images_bs={bsz}'
        for project in ['2022_07_20', '2022_07_26', '2022_08_09', '2022_08_17', '2022_08_30']:
            undist_rgb_dir = f"{project}/combined/opensfm/undistorted/images_rgb"
            thermal_final_dir = f"{project}/thermal/images_resized"
            thermal_texture_dir = f"{project}/thermal/images_resized"
            save_dir = f"{project}/combined/opensfm/undistorted/{exp_name}"
            params = {'BATCH_SIZE':bsz}
            dof = 'affine'
            map_all_images_using_NGF(undist_rgb_dir, thermal_final_dir, save_dir, thermal_texture_dir, project, params=params, thermal_ext='tiff', dof=dof)



    # # perspective
    # print("\n\n\nDDDDD")
    # exp_name = 'images_perspective'
    # for project in ['2022_07_20', '2022_07_26', '2022_08_09', '2022_08_17', '2022_08_30']:
    #     undist_rgb_dir = f"{project}/combined/opensfm/undistorted/images_rgb"
    #     thermal_final_dir = f"{project}/thermal/images_resized"
    #     thermal_texture_dir = f"{project}/thermal/images_resized"
    #     save_dir = f"{project}/combined/opensfm/undistorted/{exp_name}"
    #     params = {}
    #     dof = 'perspective'
    #     map_all_images_using_NGF(undist_rgb_dir, thermal_final_dir, save_dir, thermal_texture_dir, project, params=params, thermal_ext='tiff', dof=dof)
    #     # replace_img_names(save_dir, undist_rgb_ext, check=False)

    
    # # random vs systematic
    # print("\n\n\nWEEEE")
    # exp_name = 'images_random_select'
    # for project in ['2022_07_20', '2022_07_26', '2022_08_09', '2022_08_17', '2022_08_30']:
    #     undist_rgb_dir = f"{project}/combined/opensfm/undistorted/images_rgb"
    #     thermal_final_dir = f"{project}/thermal/images_resized"
    #     thermal_texture_dir = f"{project}/thermal/images_resized"
    #     save_dir = f"{project}/combined/opensfm/undistorted/{exp_name}"
    #     params = {'SEQUENTIAL_SELECT':False}
    #     dof = 'affine'
    #     map_all_images_using_NGF(undist_rgb_dir, thermal_final_dir, save_dir, thermal_texture_dir, project, params=params, thermal_ext='tiff', dof=dof)
    #     # replace_img_names(save_dir, undist_rgb_ext, check=False)



if __name__ == "__main__":
    run()

    # from torchviz import make_dot
    # date = '2022_08_30'
    # device = 'cpu'
    # dof = 'affine'

    # homography_net = HomographyNet(device, dof).to(device)
    # homography_net.load_state_dict(torch.load(f"{date}/ngf_homography_net.pth"))
    # H = homography_net()

    # make_dot(H, params=dict(homography_net.named_parameters()), show_attrs=True, show_saved=True).view('graphviz.gv')



    

    # # # fname = '2022_08_30/thermal/normalized/img_00000.tiff'
    # # # img_eq1 = equalize_hist(io.imread(fname)).astype(np.float32)
    # # fname = '2022_08_30/thermal/images/img_00000.tiff'
    # # img_eq = equalize_hist(io.imread(fname)).astype(np.float32)
    # # img = io.imread(fname).astype(np.float32)
    # # print(np.min(img), np.max(img))
    # # print(np.min(img_eq), np.max(img_eq))
    # # print(img_eq, img)


    # # plt.subplot(121), plt.plot(np.histogram(img_eq, bins=64)[0]), plt.imshow(img_eq, cmap='gray')
    # # plt.subplot(122), plt.plot(np.histogram(img, bins=64)[0]), plt.imshow(img, cmap='gray')
    # # plt.show()




    # rgb_img = io.imread(fname).astype(np.float32)
    # rgb_img[..., 0] = (rgb_img[..., 0] - np.min(rgb_img[..., 0])) / (np.max(rgb_img[..., 0]) - np.min(rgb_img[..., 0]))
    # rgb_img[..., 1] = (rgb_img[..., 1] - np.min(rgb_img[..., 1])) / (np.max(rgb_img[..., 1]) - np.min(rgb_img[..., 1]))
    # rgb_img[..., 2] = (rgb_img[..., 2] - np.min(rgb_img[..., 2])) / (np.max(rgb_img[..., 2]) - np.min(rgb_img[..., 2]))

    # print(histogram_mutual_information(rgb_img[:,:,0], rgb_img[:,:,1] ))
    # print(histogram_mutual_information(rgb_img[:,:,1], rgb_img[:,:,2] ))
    # print(histogram_mutual_information(rgb_img[:,:,0], rgb_img[:,:,0] ))
    pass