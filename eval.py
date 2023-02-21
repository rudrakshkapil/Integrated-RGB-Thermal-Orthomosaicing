import os
import shutil
import cv2
import numpy as np
from transform_NGF import histogram_mutual_information
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import pandas as pd
import seaborn as sns
from skimage.exposure import equalize_hist


def calculate_for_ortho(date, mode):
    # find orthos
    ortho1_path = f"{date}/orthophoto_rgb.tif"
    ortho2_path = f"{date}/orthophoto_thermal_{mode}.tif"

    # read orthos
    ortho1 = equalize_hist(cv2.imread(ortho1_path, cv2.IMREAD_GRAYSCALE))
    ortho2 = equalize_hist(cv2.imread(ortho2_path, cv2.IMREAD_GRAYSCALE)) # TODO: UNCHANGEd
    h,w = ortho1.shape
    ortho1 = cv2.resize(ortho1, (w//3,h//3))
    ortho2 = cv2.resize(ortho2, (w//3,h//3))

    # plt.figure()
    # plt.imshow(ortho1)
    # plt.figure()
    # plt.imshow(ortho2)
    # plt.show()





    # load JSON
    json_path = f'{date}/MI_results.json'
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            dict = json.load(f)
    else:
        dict = {}

    # calculate MI for each and append to list, save to dict
    MI = histogram_mutual_information(ortho1, ortho2)
    dict[mode+'_ortho'] = MI

    # save JSON
    with open(json_path, 'w') as f:
        json.dump(dict, f)
    




def calculate_over_images(date, mode):
    # find directories 
    dir1 = f"{date}/combined/opensfm/undistorted/images_rgb"
    dir2 = f"{date}/combined/opensfm/undistorted/images_{mode}"

    # read images
    imgs1 = [cv2.imread(f'{dir1}/{fname}', cv2.IMREAD_GRAYSCALE) for fname in tqdm(os.listdir(dir1), leave=False) if fname.endswith(".tif")]
    imgs2 = [cv2.imread(f'{dir2}/{fname}', cv2.IMREAD_UNCHANGED) for fname in tqdm(os.listdir(dir1), leave=False) if fname.endswith(".tif")]

    # load JSON
    json_path = f'{date}/MI_results.json'
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            dict = json.load(f)
    else:
        dict = {}


    # calculate MI for each and append to list, save to dict
    MI_list = []
    for img1, img2 in (pbar := tqdm(zip(imgs1, imgs2), leave=False, total=len(imgs1))):
        pbar.set_description('MI')
        MI = histogram_mutual_information(img1, img2)
        MI_list.append(MI)
    dict[mode+'_images_average'] = np.mean(MI_list)
    dict[mode+'_images'] = MI_list
    

    # save JSON
    with open(json_path, 'w') as f:
        json.dump(dict, f)


def eval_MI():
    # all dates with thermal, three different modes
    dates = [date for date in os.listdir('.') if date.startswith('2022')]
    modes = ['unaligned', 'manual', 'ngf']

    # loop and write to JSON files
    for date in (pbar1 := tqdm(dates[2:])):
        pbar1.set_description(f'Date: {date}')
        if '_08_03' in date:
            continue
        if 'test' in date:
            break
            
        for mode in (pbar2 := tqdm(modes, leave=False)):
            pbar2.set_description(f'Mode: {mode}')
            calculate_for_ortho(date, mode)
            # calculate_over_images(date, mode)


def checkerboard_visualize_ortho(date, mode):
    # find and load orthos
    ortho1_path = f"{date}/orthophoto_rgb.tif"
    ortho2_path = f"{date}/orthophoto_thermal_{mode}.tif"
    ortho1 = cv2.imread(ortho1_path)
    ortho2 = cv2.imread(ortho2_path)
    alpha_channel = cv2.imread(ortho1_path, cv2.IMREAD_UNCHANGED)[:,:,3]
    print(alpha_channel, alpha_channel.shape)

    # convert to tensors
    device = 'cpu'# if torch.cuda.is_available() else 'cpu'
    I_t = torch.tensor(ortho1).to(device)
    J_w = torch.tensor(ortho2).to(device)

    # interlace orthos
    Ra = I_t.clone()
    b = 300
    for i in torch.arange(0,I_t.shape[0]/b,1).int():
      for j in torch.arange(i%2,np.floor(I_t.shape[1]/b),2).int():
        Ra[i*b:(i+1)*b,j*b:(j+1)*b,:] = J_w[i*b:(i+1)*b,j*b:(j+1)*b,:].clone()


    # convert to RGB from BGR
    Ra = Ra.detach().cpu().numpy()
    Ra = cv2.cvtColor(Ra, cv2.COLOR_BGR2RGB)

    # plot
    fig, ax =plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(10)
    ax.imshow(Ra,cmap="gray")         
    plt.title(f"Orthomosaic Checkerboard - RGB and Thermal ({mode})")
    fig.set_facecolor("white")
    ax.axis('off')
    plt.show()


def checkerboard_visualize_image(date, mode, number):
    # get dirs
    rgb_dir = f"{date}/combined/opensfm/undistorted/images_rgb"
    unaligned_dir = f"{date}/combined/opensfm/undistorted/images_unaligned"
    aligned_dir = f"{date}/combined/opensfm/undistorted/images_{mode}"

    # find test image name
    img_names = [fname for fname in tqdm(os.listdir(rgb_dir), leave=False) if fname.endswith(".tif")]
    img_name = img_names[number]

    # load images
    from skimage import io
    from skimage.color import gray2rgb
    rgb_img = cv2.imread(f"{rgb_dir}/{img_name}")
    unaligned_img = cv2.imread(f"{unaligned_dir}/{img_name}")   
    aligned_img = cv2.imread(f"{aligned_dir}/{img_name}")

    min_ = max(np.min(unaligned_img), np.min(aligned_img))
    max_ = min(np.max(unaligned_img), np.max(aligned_img))
    unaligned_img = np.array((unaligned_img - min_) / (max_ - min_) * 160.0, dtype=np.uint8)
    aligned_img = np.array((aligned_img - min_) / (max_ - min_) * 160.0, dtype=np.uint8)

    # conver to torch tensors
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    I_t = torch.tensor(rgb_img).to(device)
    J_t = torch.tensor(unaligned_img).to(device)
    J_w = torch.tensor(aligned_img).to(device)

    Ra = I_t.clone()
    Rb = I_t.clone()
    b = 72
    for i in torch.arange(0,I_t.shape[0]/b,1).int():
      for j in torch.arange(i%2,np.floor(I_t.shape[1]/b),2).int():
        Rb[i*b:(i+1)*b,j*b:(j+1)*b] = J_t[i*b:(i+1)*b,j*b:(j+1)*b].clone()
        Ra[i*b:(i+1)*b,j*b:(j+1)*b] = J_w[i*b:(i+1)*b,j*b:(j+1)*b].clone()

    # convert to RGB from BGR
    Ra = Ra.detach().cpu().numpy()
    Ra = cv2.cvtColor(Ra, cv2.COLOR_BGR2RGB)
    Rb = Rb.detach().cpu().numpy()
    Rb = cv2.cvtColor(Rb, cv2.COLOR_BGR2RGB)

    # plot
    h,w,c = Ra.shape
    fig, axs = plt.subplots(1,2)
    fig.set_size_inches(14,7)
    axs[0].imshow(Rb[h//2-h//5:h//2+h//5, w//2-w//5:w//2+w//5],cmap="gray")
    axs[0].set_title("Images Before Registration")
    axs[0].axis('off')
    axs[1].imshow(Ra[h//2-h//5:h//2+h//5, w//2-w//5:w//2+w//5],cmap="gray")
    axs[1].set_title("Images After Registration")
    axs[1].axis('off')
    plt.show()


def boxplot():
    df_dict = {'Mode':[], 'MI':[]}
    dates = [date for date in os.listdir('.') if date.startswith('2022')]

    # loop and write to JSON files
    manual, ngf = [], []
    for date in ['2022_08_30']:#dates[2:]:
        if '_08_03' in date:
            continue

        with open(f"{date}/MI_results.json") as f:
            dict = json.load(f)

        modes = ['Unaligned', 'Manual', 'NGF']
        for mode in modes:
            MI_values = dict[f'{mode.lower()}_images']
            df_dict['Mode'].extend([mode]*len(MI_values))
            df_dict['MI'].extend(MI_values)
            # print(f'{mode}, {np.median(MI_values)}')

        # troubleshoot
        manual_ = np.mean(dict[f'manual_images'])
        ngf_ = np.mean(dict['ngf_images'])
        if manual_ > ngf_:
            manual.append(date + f' - {str(manual_-ngf_)}')
        else:
            ngf.append(date + f' - {str(ngf_-manual_)}')
    df = pd.DataFrame(df_dict)

    # plot
    plt.figure(figsize=(7,5))
    # sns.set_style("darkgrid")
    palette = sns.color_palette("Set2", n_colors=len(modes))
    palette.reverse()
    sns.set_palette(palette)
    ax = sns.boxplot(data=df, x="Mode", y="MI", showmeans=True, meanprops={"marker":"o",
                        "markerfacecolor":"white", 
                        "markeredgecolor":"black",
                        "markersize":"10"})
    ax.set(title="MI Distribution using Different Registration Techniques")
    ax.set(xlabel="Image Registration Technique for Thermal Images", ylabel="Histogram-based MI with Undistorted RGB Images")
    ax.set(xticklabels = ['Unregistered', 'Manual', 'NGF'])
    plt.show()

    # print averages
    print(df.groupby('Mode').median())



    
def visualize_orthos(mode):
    dates = [date for date in os.listdir('.') if date.startswith('2022')]
    for date in dates[2:]:
        print(date)
        if '08_03' in date: 
            continue
        ortho_path = f"{date}/orthophoto_thermal_{mode}.tif"
        ortho = cv2.imread(ortho_path, cv2.IMREAD_UNCHANGED)

        figure = plt.figure(figsize=(20,10))
        plt.title(date)
        plt.imshow(ortho, cmap='gray')
        plt.xlim(3500, 4500)
        plt.ylim(4000,3000)
        plt.show()



def visualize_images_vs_ortho_hists():
    path = '2022_08_30/rgb/opensfm/reconstruction.json'
    with open(path, 'r') as f:
        d = json.load(f)
    
    # get shot names
    shots = d[0]['shots']
    shot_names = [f'{key.split(".")[0]}.JPG.tif' for key,val in shots.items()]
    

    translations = np.array([shot['translation'][0:2] for id,shot in shots.items()])
    max_ = np.max(translations, axis=0)
    min_ = np.min(translations, axis=0)
    print(max_, min_)

    # gps -> working for 656
    # translations = np.array([shot['gps_position'][0:2] for id,shot in shots.items()])
    # max_ = np.max(translations, axis=0)
    # min_ = np.min(translations, axis=0)
    # print(max_, min_)
    # exit(0)

    # load image
    img_idx = shot_names.index('img_00502.JPG.tif')
    print(shot_names[img_idx])
    plt.figure(figsize=(18,4))
    # dir_path = '2022_08_30/combined/opensfm/undistorted/images_rgb'
    # img = cv2.imread(f'{dir_path}/{shot_names[img_idx]}')
    dir_path = '2022_08_30/combined/opensfm/undistorted/images_ngf_norm'
    img = cv2.imread(f'{dir_path}/{shot_names[img_idx]}', cv2.IMREAD_ANYDEPTH)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) # switch
    print(img)
    h,w,*_ = img.shape
    img_y, img_x = translations[img_idx]
    print(translations[img_idx])
    # plt.subplot(131),plt.imshow(cv2.rotate(img[:,:,::-1], cv2.ROTATE_90_COUNTERCLOCKWISE)), plt.title('Original Image'),plt.axis('off')
    
    img = img.astype(np.float32)/256.0
    img = np.array(img, dtype=np.uint8)
    img -= 0
    plt.subplot(131),plt.imshow(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), cmap='gray'), plt.title('Original Image'),plt.axis('off')

    # load ortho
    # ortho_path = '2022_08_30/output/orthophoto_combined_rgb.tif'
    # ortho = cv2.imread(ortho_path, cv2.IMREAD_UNCHANGED)[:,:,:3]
    ortho_path = '2022_08_30/output/orthophoto_combined_thermal_NGF.tif'
    ortho = cv2.imread(ortho_path, cv2.IMREAD_ANYDEPTH)
    
    hh, ww, *_ = ortho.shape

    # get location (x,y)
    x_scale = (max_[1]-min_[1])
    y_scale = (max_[0]-min_[0])
    y = int((1-((img_y - min_[0]) / y_scale)) * hh)
    x = int((((img_x - min_[1]) / x_scale)) * ww)
    print(x,y)
    # exit(0)

    # adjust size - working for RGB
    scale = 2.5
    h = int(h/scale)
    w = int(w/scale)
    shift_y = -int(h/1.8)  # --- working for 00656
    shift_x = -int(w/1.8) # 

    # scale = 2.65
    # h = int(h/scale)
    # w = int(w/scale)
    # shift_y = -80#-int(h/15)  # --- working for 00656
    # shift_x = -int(w/2)-30 # 

    # extracted_patch = ortho[y+shift_y : y+h+shift_y, x+shift_x : x+w+shift_x,::-1]
    # plt.subplot(132),plt.imshow(cv2.rotate(extracted_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)), plt.title('Extracted Patch from Orthomosaic Image'),plt.axis('off')

    extracted_patch = ortho[y+shift_y : y+h+shift_y, x+shift_x : x+w+shift_x]
    # extracted_patch = extracted_patch.astype(np.float32)*256.0 # maybe remove
    # extracted_patch = np.array(extracted_patch, dtype=np.uint16)

    plt.subplot(132),plt.imshow(cv2.rotate(extracted_patch, cv2.ROTATE_90_COUNTERCLOCKWISE), cmap='gray'), plt.title('Extracted Patch from Orthomosaic Image'),plt.axis('off')

    # plt.figure(),plt.imshow(ortho[...,::-1])
    # plt.show()



    
    


    # plot histograms (ignore all black 0s)
    # h,w = extracted_patch.shape[:2]
    # img = cv2.resize(img, (w,h))
    # print(img, extracted_patch)
    hist1,bins1 = np.histogram(img, bins=255, range=(1,256))
    hist2,bins2 = np.histogram(extracted_patch, bins=255, range=(1,256))
    # hist1,bins1 = np.histogram(img, bins=65535, range=(1,65536))
    # hist2,bins2 = np.histogram(extracted_patch, bins=65535, range=(1,65536))
    hist1 = np.array(hist1.astype(np.float32)/np.sum(hist1), dtype=np.float32)
    hist2 = np.array(hist2.astype(np.float32)/np.sum(hist2), dtype=np.float32)
    plt.subplot(133)
    plt.title('Image Histograms (for pixel values > 0)')
    plt.plot(hist2, 'purple', label = 'Extracted Patch from Ortho')
    plt.plot(hist1, 'orange', label = 'Original Image')
    
    plt.xlabel('Pixel Value')
    plt.ylabel('Normalized Frequency')
    plt.legend()
    plt.show() 

    bc = sum(np.sqrt(np.multiply(hist1, hist2)))
    print(bc)

    cd = np.dot(hist1,hist2)/(np.linalg.norm(hist1)*np.linalg.norm(hist2))
    print(cd)




# checkerboard_visualize_ortho('2022_08_30', 'unaligned')
# checkerboard_visualize_image('2022_08_30', 'ngf', 740)
# boxplot()

# visualize_orthos('ngf')

# eval_MI()

# boxplot()

# checkerboard_visualize_image('2022_08_30', 'NGF', 187) # 187 is good
visualize_images_vs_ortho_hists()