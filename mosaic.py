from doctest import testfile
import io
import os
import shutil
import sys
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
import yaml
import torch
from matplotlib import gridspec 
import skimage

from skimage.util.dtype import dtype_range
from skimage.morphology import disk
from skimage.filters import rank
from skimage import img_as_ubyte
from skimage import exposure
from skimage import io 

import rawpy 


from utm_converter import utm
from transform_NGF import map_all_images_using_NGF

from df_repo.deepforest import main

DEBUG = False


## ------------ RGB MAPPING ----------- ##

def print_exif(img):
    exif = { TAGS[k]: v for k, v in img._getexif().items() if k in TAGS }
    print(exif.keys()) # list of exif metadata exists in the image

    geoexif = {}
    for (tag, value) in GPSTAGS.items():
        if tag in exif["GPSInfo"]:
            geoexif[value] = exif['GPSInfo'][tag]

    print(geoexif.items()) # list of GPS information exists in the image
    print("GeoEXIF printed")
        
def prepare_mosaic_images(dir, save_dir, crop=True, scale=5, overwrite=False, ext="_W.JPG"):
    '''
    Copy images from dir to save_dir after cropping center portion
    Exif data is also specifically copied over 
    '''
    img_names = [fname for fname in os.listdir(dir) if fname.endswith(ext)]


    # create save_dir if not there
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # loop
    for img_name in tqdm(img_names, file=sys.stdout):
        # get exif data
        PIL_img = Image.open(f"{dir}/{img_name}")
        exif = PIL_img.info['exif']
        
        # crop center portion 
        np_img = np.array(PIL_img)
        if crop:
            h,w,c = np_img.shape
            ymin = int(h//2-h//scale)
            ymax = int(h//2+h//scale)
            xmin = int(w//2-w//scale)
            xmax = int(w//2+w//scale)

            np_img = np_img[ymin:ymax, xmin:xmax]

        # Convert OpenCV image onto PIL Image
        PIL_img_cropped = Image.fromarray(np_img)

        # Encode newly-created image into memory as JPEG along with EXIF from other image
        save_path = f"{save_dir}/{img_name}"
        PIL_img_cropped.save(save_path, format='JPEG', exif=exif) 
        PIL_img.close()




## ------------ THERMAL MAPPING ----------- ##

def prepare_thermal_images(dir, save_dir, ext="_T.JPG"):
    '''
    Copy thermal images from dir to save_dir
    No need to crop so can copy over directly (exif data automatically handled)
    '''
    img_names = [fname for fname in os.listdir(dir) if fname.endswith(ext)]
    if DEBUG:
        img_names = img_names[:20]

    # create save_dir if not there
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # # read one image and check size
    # l_ = len(cv2.imread(f'{dir}/{img_names[0]}', cv2.IMREAD_UNCHANGED).shape)
    # print(l_)
    # if l_ == 2:
    # loop over images and copy over
    for img_name in tqdm(img_names):
        src_path = f"{dir}/{img_name}"
        dst_path = f"{save_dir}/{img_name}"
        shutil.copy(src=src_path, dst=dst_path)
    # elif l_ == 3:
    #     # loop over images and copy over
    #     for img_name in tqdm(img_names):
    #         im = cv2.imread(f'{dir}/{img_name}')
    #         # im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    #         cv2.imwrite(f"{save_dir}/{img_name}", im)



def remove_original_tiffs(tiff_dir):
    names = os.listdir(tiff_dir)
    count = 0
    for name in names:
        if name.endswith("_original"):
            os.remove(f"{tiff_dir}/{name}") 
            count += 1
    print("Removed ", count)

def sort_tiff_names(tiff_names):
    tiff_names_temp = []
    for name in tiff_names:
        ext = name.split('.')[-1]
        name_without_ext = os.path.splitext(name)[0]

        prefix = "_".join(name_without_ext.split('_')[:-1])
        index = name_without_ext.split('_')[-1]
        index = index.zfill(4) # NOTE: increase this to 5 if you have > 9999 images 

        tiff_names_temp.append(f'{prefix}_{index}.{ext}')
    

    sorted_indices = sorted(range(len(tiff_names_temp)), key=lambda k: tiff_names_temp[k])
    return sorted_indices

def insert_thermal_tiff_exifs(rjpeg_dir, tiff_dir, log_file):
    '''
    Take exif from rjpeg and insert into corresponding tif
    '''
    remove_original_tiffs(tiff_dir)
    rjpeg_names = os.listdir(rjpeg_dir)
    tiff_names = os.listdir(tiff_dir)
    sorted_indices = sort_tiff_names(tiff_names)
    tiff_names = [tiff_names[i] for i in sorted_indices]

    if not os.path.exists(tiff_dir):
        os.makedirs(tiff_dir)

    for i in tqdm(range(len(tiff_names))):
        src_path = f"{rjpeg_dir}/{rjpeg_names[i]}"
        dst_path = f"{tiff_dir}/{tiff_names[i]}"
        call = f"exiftool -tagsfromfile {src_path} \"-exif:all>exif:all\" {dst_path} >> {log_file} 2>&1" # pipe stderr (2) andstdout (1) to output file
        os.system(call)

    remove_original_tiffs(tiff_dir)


        

def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins)
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')

    xmin, xmax = dtype_range[image.dtype.type]
    ax_hist.set_xlim(xmin, xmax)

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')

    return ax_img, ax_hist, ax_cdf


def local_hist_eq(img_dir, save_dir, resize=False, new_size=True):
    # load images in RGB format
    img_names = os.listdir(img_dir)
    imgs = np.asarray([cv2.imread(f"{img_dir}/{img_name}", cv2.IMREAD_UNCHANGED) for img_name in img_names])
    if DEBUG:
        imgs = imgs[:20]

    # create directory if needed
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # min max normalize and save
    for idx,img in tqdm(enumerate(imgs), total=len(imgs)):
        ## local histogram eq -- not so good maybe
        # img = img_as_ubyte(img) ## -> for making it 8 bit (not a good idea)
        # footprint = disk(150)
        # new_img = rank.equalize(img, footprint=footprint)

        ## contrast stretching
        # p2, p98 = np.percentile(img, (2, 98))
        # new_img = exposure.rescale_intensity(img, in_range=(p2, p98))

        ## global equalization (image)
        new_img = exposure.equalize_hist(img, nbins=2e16)
        new_img = np.array(new_img*(2e8-1), dtype=np.uint16)


        cv2.imwrite(f"{save_dir}/{img_names[idx]}", new_img) 


def min_max_norm_images(img_dir, save_dir, log_file, resize=False, size=None):
    # load images in RGB format
    img_names = os.listdir(img_dir)
    imgs = np.asarray([cv2.imread(f"{img_dir}/{img_name}", cv2.IMREAD_UNCHANGED) for img_name in img_names]) 
    print(imgs.shape)


    # find min and max values in entire image dir  
    min_ = np.min(imgs)
    max_ = np.max(imgs)
    with open(log_file, "w") as f:
        f.write(f"min:{min_}\nmax:{max_}\n")

    # create directory if needed
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # min max normalize and save 
    for idx,img in tqdm(enumerate(imgs), total=len(imgs)):
        norm_img = (img - min_) / (max_ - min_)
        norm_img = np.array(norm_img * 65535, dtype=np.uint16) # 16 bit image => 2^16 - 1 = 65535
        if resize:
            norm_img = cv2.resize(norm_img, size, cv2.INTER_CUBIC)      
        cv2.imwrite(f"{save_dir}/{img_names[idx]}", norm_img) 

        


def convert_rjpeg_to_raw(exec_path, rjpeg_dir, thermal_dir, log_file, function="measure", humidity=70, emissivity=0.95, distance=25, reflection=25):
    # call SDK
    options = f'--emissivity {emissivity} --humidity {humidity} --distance {distance} --reflection {reflection} --verbose detail'
    if function == 'measure':
        call =  f"{exec_path} -s {rjpeg_dir} -a measure -o {thermal_dir}/img --measurefmt float32 {options} > {log_file}"
    else:
        call = f"{exec_path} -s {rjpeg_dir} -a {function} -o {thermal_dir}/img {options} > {log_file}"
    os.system(call)

    # create save dir
    if not os.path.exists(f"{thermal_dir}/raw"):
        os.makedirs(f"{thermal_dir}/raw")

    # move all raw to raw_dir
    raw_names = [fname for fname in os.listdir(thermal_dir) if fname.endswith('.raw')]
    for raw_name in tqdm(raw_names):
        src_path = f"{thermal_dir}/{raw_name}"
        dst_path = f"{thermal_dir}/raw/{raw_name}"
        shutil.move(src=src_path, dst=dst_path)

    # check if all images worked
    before = len(os.listdir(rjpeg_dir))
    after = len(os.listdir(f"{thermal_dir}/raw"))
    assert before == after, "DJI SDK Issue! Not all images converted due to chosen settings. Change and retry."


from PIL import Image
def rasterize_raw_to_tiffs(raw_dir, tiff_dir, ROOT_DIR, function='extract', img_size = (512,640)):
    # load images in RGB format
    img_names = [f"{raw_dir}/{fname}" for fname in os.listdir(raw_dir)]
    sorted_indices = sort_tiff_names(img_names)
    img_names = [img_names[i] for i in sorted_indices]

    if not os.path.exists(tiff_dir):
        os.makedirs(tiff_dir)

    for idx, raw_name in enumerate(tqdm(img_names)):
        img = np.fromfile(raw_name, dtype=np.float32)
        img = img.reshape(img_size[0],img_size[1])
    
        out_path = f'{tiff_dir}/img_{idx}.tiff'
        if idx == 0:
            io.imshow(img)
        io.imsave(out_path, img)
    
    # R_EXE_PATH = f'{ROOT_DIR}/R/R-4.2.1/bin/Rscript.exe'

    # # create save dir
    # if not os.path.exists(tiff_dir):
    #     os.makedirs(tiff_dir)

    # # call R script with arguments
    # r_script = 'rasterize.R'# if function == 'extract' else 'rasterize_temp.R'
    # num_images = len(os.listdir(raw_dir))
    # call = f"{R_EXE_PATH} rasterize.R {ROOT_DIR}/{raw_dir}/img_ {ROOT_DIR}/{tiff_dir}/img_ {num_images}"
    # os.system(call)


def resize_images(img_dir, save_dir, size):
    # load images in RGB format
    img_names = os.listdir(img_dir) 
    imgs = np.asarray([cv2.imread(f"{img_dir}/{img_name}", cv2.IMREAD_UNCHANGED) for img_name in img_names])

    # create directory if needed
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # min max normalize and save 
    for idx,img in tqdm(enumerate(imgs), total=len(imgs)):
        img = cv2.resize(img, size, cv2.INTER_CUBIC)
        cv2.imwrite(f"{save_dir}/{img_names[idx]}", img) 



def replace_img_names(dir, ext='jpeg', check=False):
    '''
    Rename images in dir as img_00001.<ext>, img_00002.<ext>, etc.
    '''
    img_names = os.listdir(dir)
    counter = 0
    if check:
        check_dir = dir+'_rgb'
        check_list = [fname.split('.')[0] for fname in os.listdir(check_dir)]
    
    for img_name in img_names:
        src = f"{dir}/{img_name}"
        dst = f"{dir}/img_{str(counter).zfill(5)}.{ext}"
        counter += 1

        if check and img_name.split('.')[0] not in check_list:
            print(f"Skipping {img_name}")
            os.remove(src)
        else:
            shutil.move(src, dst)
        

def rename_tiff_with_leading_zeros(dir, ext='tiff', zeros=5):
    img_names = os.listdir(dir)

    for name in img_names:
        num = name.split('_')[1].split('.')[0]
        save_img_name = f"img_{num.zfill(zeros)}.{ext}"
        dst = f"{dir}/{save_img_name}"
        shutil.move(f"{dir}/{name}", dst)
    




def denormalize_ortho(img_path, save_path, norm_log_file):
    # read image
    norm_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    # read min, max
    with open(norm_log_file, "r") as f:
        data = f.readlines()
        min_ = int(data[0].strip().split(':')[1])
        max_ = int(data[1].strip().split(':')[1])

    # norm_img = (img - min_) / (max_ - min_)
    img = norm_img * (max_ - min_) + min_

    # save
    cv2.imwrite(save_path, img)

    # copy over exif
    call = f"exiftool -tagsfromfile {img_path} \"-exif:all>exif:all\" {save_path}" # pipe stderr (2) andstdout (1) to output file
    os.system(call)
    os.remove(save_path+'_original')









## ------------ DATA CONVERSION --------- ##
    


def conversion(old):
    direction = {'N':1, 'S':-1, 'E': 1, 'W':-1}
    new = old.replace(u'°',' ').replace('\'',' ').replace('"',' ')
    new = new.split()
    new_dir = new.pop()
    new.extend([0,0,0])
    return (int(new[0])+int(new[1])/60.0+int(new[2])/3600.0) * direction[new_dir]


def copy_subset(dir, save_dir, count=10):
    img_names = os.listdir(dir)
    img_names = img_names[:count]

    # create save_dir if not there
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # loop over images and copy over
    for img_name in tqdm(img_names):
        src_path = f"{dir}/{img_name}"
        dst_path = f"{save_dir}/{img_name}"
        shutil.copy(src=src_path, dst=dst_path)

            

def map_all_images_manually(project, rgb_dir, thermal_dir, save_dir, thermal_texture_dir, img_index, eq_hist=False, dof='affine'):

    # get image names
    rgb_img_paths = [f"{rgb_dir}/{fname}" for fname in os.listdir(rgb_dir) if os.path.isfile(f"{rgb_dir}/{fname}")]
    thermal_img_paths = [f"{thermal_dir}/{fname}" for fname in os.listdir(thermal_dir) if f"{thermal_dir}/{fname}"]

    # loop till satisfied with homography
    with torch.no_grad():
        while True:
            rgb_img_path = rgb_img_paths[img_index]
            thermal_img_path = thermal_img_paths[img_index]

            # load images, get shapes
            thermal_img = skimage.io.imread(thermal_img_path)
            rgb_undistort_img = skimage.io.imread(rgb_img_path)
            hh,ww,_ = rgb_undistort_img.shape

            if thermal_img.shape != rgb_undistort_img.shape:
                print("Resizing thermal image")
                thermal_img = skimage.transform.resize(thermal_img, (hh,ww))

            # hist eq for easier visualization
            if eq_hist:
                print(thermal_img)
                thermal_img = img_as_ubyte(exposure.equalize_hist(thermal_img))
                print(thermal_img)
            else:
                print(thermal_img)
                min_ = np.amin(thermal_img)
                max_ = np.amax(thermal_img)
                thermal_img = (thermal_img - min_) / (max_ - min_) * 255.0
                thermal_img = thermal_img.astype(np.uint8)
                print(thermal_img)
                

            num_pts = 4 if dof == 'perspective' else 3
            # Get point correspondences(homography)
            fig1 = plt.figure(figsize=(8,6))
            plt.title(f"Select {num_pts} points on this Thermal Image")
            plt.imshow(thermal_img[:,:], cmap='gray')
            pts2 = np.float32(plt.ginput(num_pts))
            

            fig2 = plt.figure(figsize=(8,6))
            plt.title(f"Select the same {num_pts} points on this RGB Image")
            plt.imshow(rgb_undistort_img)
            pts1 = np.float32(plt.ginput(num_pts))
            plt.close(fig2)
            plt.close(fig1)

            ## calculate transformation matrix
            if dof == 'perspective':
                M = cv2.getPerspectiveTransform(pts1,pts2)
                projected = cv2.warpPerspective(thermal_img, M, (ww,hh), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            else:
                M = cv2.getAffineTransform(pts1, pts2)
                projected = cv2.warpAffine(thermal_img, M, (ww,hh), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

            # conver to torch tensors
            device = 'cpu' #'cuda:0' if torch.cuda.is_available() else 'cpu'
            # rgb_undistort_img = cv2.cvtColor(rgb_undistort_img, cv2.COLOR_RGB2GRAY)
            I_t = torch.tensor(rgb_undistort_img).to(device)
            J_t = torch.tensor(thermal_img).to(device)
            J_w = torch.tensor(projected).to(device)

            Ra = I_t.clone()
            Rb = I_t.clone()
            b = 120
            for i in torch.arange(0,I_t.shape[0]/b,1).int():
                for j in torch.arange(i%2,np.floor(I_t.shape[1]/b),2).int():
                    Rb[i*b:(i+1)*b,j*b:(j+1)*b] = J_t[i*b:(i+1)*b,j*b:(j+1)*b].clone().unsqueeze(-1)
                    Ra[i*b:(i+1)*b,j*b:(j+1)*b] = J_w[i*b:(i+1)*b,j*b:(j+1)*b].clone().unsqueeze(-1)

            print(Ra.shape, Rb.shape)

            # convert to RGB from BGR
            Ra = Ra.detach().cpu().numpy()
            # Ra = cv2.cvtColor(Ra, cv2.COLOR_BGR2RGB)
            Rb = Rb.detach().cpu().numpy()
            # Rb = cv2.cvtColor(Rb, cv2.COLOR_BGR2RGB)

            # plot
            fig, axs = plt.subplots(2,2)
            fig.set_size_inches(14,7)
            plt.title('Manual Registration Performance. Check console for next steps.')
            axs[0,0].imshow(thermal_img, cmap='gray')
            axs[0,0].set_title("Image Before Registration")
            axs[0,0].axis('off')
            axs[0,1].imshow(J_w.detach().cpu().numpy(), cmap='gray')
            axs[0,1].set_title("Image After Registration")
            axs[0,1].axis('off')
            axs[1,0].imshow(Rb)
            axs[1,0].set_title("Checkerboard Before Registration")
            axs[1,0].axis('off')
            axs[1,1].imshow(Ra)
            axs[1,1].set_title("Checkerboard After Registration")
            axs[1,1].axis('off')
            plt.show()

            # get user input for alignmnet
            u_input = ""
            while u_input not in ['Y', 'N', 'S']:
                u_input = input("Are you satisfied with the alignment?\nY => yes, N => Try Again, S => Switch images\n > ")
                
            if u_input == 'Y':
                break
            elif u_input == 'S':
                img_index = -1
                while img_index < 0 or img_index >= len(rgb_img_paths):
                    img_index = input(f"Please enter a new valid image index from {rgb_dir} to use.\nOptions: 0 - {len(rgb_img_paths)-1} > ")
                    if not img_index.isnumeric():
                        img_index = -1
                    else: img_index = int(img_index)

    log_homography_file = f"{project}/logs/log_manual_homography.txt"# f"{date}/log_homography.txt"
    print("Calculated M =>", M)
    with open(log_homography_file, "w") as f:
        f.write(str(M))

   


    

    ## Do it for all images
    # create save_dir directory if needed
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # get images to transform in list
    img_names = os.listdir(thermal_texture_dir)
    thermal_imgs = [cv2.imread(f"{thermal_texture_dir}/{img_name}", cv2.IMREAD_UNCHANGED) for img_name in img_names]

    # loop over list and warp
    for idx, thermal_img in tqdm(enumerate(thermal_imgs), total=len(thermal_imgs)):
        if dof == 'affine':
            projected_img = cv2.warpAffine(thermal_img, M, (ww,hh), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        else:
            projected_img = cv2.warpPerspective(thermal_img, M, (ww,hh), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        cv2.imwrite(f"{save_dir}/{img_names[idx]}", projected_img) 
    

def get_ODM_flags(ODM_params):
    flags = ''
    for key, value in ODM_params.items():
        flags += f'--{key} {value} '
    return flags

def get_directories(cfg):
    # root / project. Also create log dir
    project = cfg["DIR"]["PROJECT"] 
    if not os.path.exists(f"{project}/logs"):
        os.makedirs(f"{project}/logs")

    # RGB only
    rgb_dir = cfg["DIR"]["RGB"]

    # Thermal only
    thermal_dir = cfg["DIR"]["THERMAL"]

    # mapping data (combined images for H20T) -- maybe nested (mapping_data/folder1/..., mapping_data/folder2/..., etc.)
    mapping_dir = cfg["DIR"]["H20T_DATA"]
    if mapping_dir == 'auto':
        mapping_dir = f"{project}/mapping_data"
    if mapping_dir == '':
        mapping_dir = None
    nested = None
    if mapping_dir is not None:
        nested = True
        for name in os.listdir(mapping_dir):
            if not os.path.isdir(f"{mapping_dir}/{name}"):
                nested = False
                break
        
    if rgb_dir != '' and thermal_dir != '':
        mapping_dir = None

    # RGB only
    rgb_dir = cfg["DIR"]["RGB"]

    # Thermal only
    thermal_dir = cfg["DIR"]["THERMAL"]

    # output
    output_dir = f"{project}/output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return project, mapping_dir, nested, rgb_dir, thermal_dir, output_dir


def determine_extensions(camera, rgb_dir, thermal_dir):
    '''
    Find and return the extensions of the rgb and thermal images (unprocessed)
    '''
    if camera == "H20T":
        return 'JPG', 'tiff'

    rgb_ext = os.listdir(rgb_dir)[0].split('.')[-1]
    thermal_ext = os.listdir(thermal_dir)[0].split('.')[-1]
    return rgb_ext, thermal_ext
    


## ------------ PIPELINE WRAPPERS ----------- ##



def combined_mapping_pipeline(cfg):
    """
    Create RGB orthomosaic from mapping data.
    Then, create a copy and texture using thermal images.
    """
    # set up directories and extract info
    project, mapping_dir, nested, rgb_dir, thermal_dir, output_dir = get_directories(cfg)
    camera = cfg["CAMERA"]
    rgb_ext, _ = determine_extensions(camera, rgb_dir, thermal_dir)
    optimize_disk_space = cfg["OUTPUT"]["OPTIMIZE_DISK_SPACE"]
    ROOT_DIR = os.path.abspath(os.getcwd()).replace('\\','/')

    ## STAGE 1 -- RGB ORTHO
    if cfg["STAGES"]['COMBINED_STAGES']['STAGE_1']:
        print("\n\n------------ STAGE 1 ------------")

        # extract options
        crop = cfg['PREPROCESSING']['RGB']['CROP']
        scale = cfg['PREPROCESSING']['RGB']['SCALE']

        # remove img dir:
        try: shutil.rmtree(f'{project}/rgb/images')
        except: pass
        
        # move all RGB images to single directory (also crop and copy over EXIFs properly)
        if camera == "H20T" and mapping_dir is not None:
            print("Moving RGB images from H20T Mapping Data")
            if nested:
                inner_dirs = os.listdir(mapping_dir)
                for dir in inner_dirs:
                    prepare_mosaic_images(f'{mapping_dir}/{dir}', f'{project}/rgb/images', crop=crop, scale=scale, overwrite=True, ext="_W.JPG")
            else:
                prepare_mosaic_images(f'{mapping_dir}', f'{project}/rgb/images', crop=crop, scale=scale, overwrite=True, ext="_W.JPG")
        else:
            prepare_mosaic_images(f'{rgb_dir}', f'{project}/rgb/images', crop=crop, scale=scale, overwrite=True, ext="")
        replace_img_names(f'{project}/rgb/images', ext=rgb_ext)

        # run ODM on RGB data
        os.chdir("ODM")
        flags = get_ODM_flags(cfg["ODM"])
        second_part = f"{ROOT_DIR}/{project}/rgb" if ROOT_DIR not in project else f"{project}/rgb"
        call_1 = f".\\run.bat {flags} {second_part}"
        os.system(call_1)
        os.chdir("..")


        # Move output
        print("Moving RGB ortho and copying intermediate results...")
        shutil.move(f'{project}/rgb/odm_orthophoto/odm_orthophoto.tif', f'{output_dir}/orthophoto_combined_rgb.tif') 

        # copy RGB intermediate results to project/combined 
        try: shutil.rmtree(f"{project}/combined")
        except: pass
        shutil.copytree(f"{project}/rgb", f"{project}/combined")
        shutil.move(f"{project}/combined/images", f"{project}/combined/images_rgb")
        shutil.move(f"{project}/combined/opensfm/undistorted/images", f"{project}/combined/opensfm/undistorted/images_rgb")

        if optimize_disk_space:
            shutil.rmtree(f"{project}/rgb")
            shutil.rmtree(f"{project}/combined/images_rgb")



    ### STAGE 2 -- THERMAL PREPROCESSING (only for H20T drone)
    if cfg["STAGES"]['COMBINED_STAGES']['STAGE_2']: 
        print("\n\n------------ STAGE 2 ------------")
        try: shutil.rmtree(f'{project}/thermal') 
        except: pass

        if camera == "H20T":
            # Move RJPEGs to one directory
            print("Moving RJPEGs to single directory...")
            if mapping_dir is not None:
                if nested:
                    inner_dirs = os.listdir(f"{mapping_dir}")
                    for dir in inner_dirs:
                        prepare_thermal_images(f'{mapping_dir}/{dir}', f'{project}/thermal/rjpegs', ext="_T.JPG")
                else:
                    prepare_thermal_images(f'{mapping_dir}', f'{project}/thermal/rjpegs', ext="_T.JPG")
            else:
                prepare_thermal_images(f'{thermal_dir}', f'{project}/thermal/rjpegs', ext="")
            
            # Convert RJPEGs to raw binary files / temp files
            print("Converting rjpegs to raw...")
            function = cfg["PREPROCESSING"]["THERMAL"]["H20T"]["FUNCTION"]
            exec_path = None # cfg["PREPROCESSING"]["THERMAL"]["H20T"]["DJI_THERMAL_SDK_PATH"]
            if exec_path is None or (not os.path.exists(exec_path)):
                exec_path = f".\\DJI_Thermal_SDK\\utility\\bin\\windows\\release_x64\\dji_irp_omp.exe"
            print(os.path.exists(exec_path))
            humidity = cfg["PREPROCESSING"]["THERMAL"]["H20T"]["HUMIDITY"]
            distance = cfg["PREPROCESSING"]["THERMAL"]["H20T"]["DISTANCE"]
            reflection = cfg["PREPROCESSING"]["THERMAL"]["H20T"]["TEMPERATURE"]
            emissivity = cfg["PREPROCESSING"]["THERMAL"]["H20T"]["EMISSIVITY"]
            convert_rjpeg_to_raw(exec_path, f'{project}/thermal/rjpegs', f'{project}/thermal', f"{project}/logs/log_binary-extraction.txt", function=function, humidity=humidity, emissivity=emissivity, distance=distance, reflection=reflection)


            ## rasterize binary data to TIFFs
            print("Converting raw to tiffs...")
            rasterize_raw_to_tiffs(f'{project}/thermal/raw', f'{project}/thermal/images', ROOT_DIR, function) 
        

        # for other drones, just copy over data to project/thermal/images (assuming preprocessed)
        else:
            print("Moving images from provided thermal directory to thermal/images ...")
            prepare_thermal_images(thermal_dir, f'{project}/thermal/images', ext="")
    _, thermal_ext = determine_extensions(camera, rgb_dir, f'{project}/thermal/images')
    

    ### STAGE 3 -- RGB THERMAL ALIGNMENT
    if cfg["STAGES"]['COMBINED_STAGES']['STAGE_3']:
        print("\n\n------------ STAGE 3 ------------")

        # check if previous two steps have generated their needed outputs: (and get size)
        undist_rgb_dir = f"{project}/combined/opensfm/undistorted/images_rgb"
        undist_rgb_ext = ".".join(os.listdir(undist_rgb_dir)[1].split('.')[1:])
        if not os.path.exists(undist_rgb_dir):
            print("Uh oh! Something may have gone wrong with Stage 1. Please try again.")
            print("Details => undistorted images not present (likely due to failed RGB orthomosaicing)")
            return
        if not os.path.exists(f'{project}/thermal/images'):
            print("Uh oh! Something may have gone wrong with Stage 2. Please try again.")
            print("Details => thermal/images directory doesn't exist. Check DIR.H20T data or DIR.THERMAL in config file.")
            return

        ## normalize
        # rename 
        thermal_final_dir = f"{project}/thermal/images"
        rename_tiff_with_leading_zeros(thermal_final_dir, ext=thermal_ext)
        replace_img_names(thermal_final_dir, ext=thermal_ext)

        # get rgb size
        undist_rgb_img_path = f"{undist_rgb_dir}/{os.listdir(undist_rgb_dir)[1]}"
        undist_rgb_img = cv2.imread(undist_rgb_img_path)
        h,w,_ = undist_rgb_img.shape

        # normalize (not by default)
        if cfg["PREPROCESSING"]["THERMAL"]["NORMALIZE"]:
            print("Normalizing...")
            thermal_final_dir = f"{project}/thermal/normalized"
            try: shutil.rmtree(thermal_final_dir)
            except: pass            
            min_max_norm_images(f'{project}/thermal/images', thermal_final_dir, f"{project}/logs/log_normalization.txt", resize=True, size=(w,h))

            # prepare textures (correct size) -- check if need to denormalize
            if cfg["OUTPUT"]["DENORMALIZE"]:
                resize_images(f'{project}/thermal/images', f'{project}/thermal/images_resized', size=(w,h))
                thermal_texture_dir = f"{project}/thermal/images_resized"
            else:
                thermal_texture_dir = f"{project}/thermal/normalized"
        else:
            thermal_final_dir = f"{project}/thermal/images_resized"
            print("Resizing thermal images to RGB size...") 
            resize_images(f'{project}/thermal/images', f'{project}/thermal/images_resized', size=(w,h))
            thermal_texture_dir = f"{project}/thermal/images_resized"

        # do registration - determine mode first   
        mode = cfg["HOMOGRAPHY"]["MODE"]
        save_dir = f"{project}/combined/opensfm/undistorted/images"
        try: shutil.rmtree(save_dir)
        except: pass
        print("Homography calculation...")
        dof = cfg["HOMOGRAPHY"]["DOF"]
        if mode == "MANUAL":
            ## Homography computation from manual point correspondences
            img_index = cfg["HOMOGRAPHY"]["MANUAL_PARAMS"]["IMAGE_PAIR_INDEX"]
            hist_eq = cfg["HOMOGRAPHY"]["MANUAL_PARAMS"]["HIST_EQ"]
            map_all_images_manually(project, undist_rgb_dir, thermal_final_dir, save_dir, thermal_texture_dir, img_index, eq_hist=hist_eq, dof=dof)
            replace_img_names(save_dir, undist_rgb_ext, check=False)

        elif mode =="NGF":
            ## NGF to find homography
            NGF_params = cfg["HOMOGRAPHY"]["NGF_PARAMS"]
            map_all_images_using_NGF(undist_rgb_dir, thermal_final_dir, save_dir, thermal_texture_dir, project, params=NGF_params, thermal_ext=thermal_ext, dof=dof)
            replace_img_names(save_dir, undist_rgb_ext, check=False)

        elif mode =="ECC":
            ## ECC to find homography (same as above, just diff loss function)
            ECC_params = cfg["HOMOGRAPHY"]["NGF_PARAMS"]
            ECC_params['NGF_FLAG'] = False # only need to change this
            map_all_images_using_NGF(undist_rgb_dir, thermal_final_dir, save_dir, thermal_texture_dir, project, params=ECC_params, thermal_ext=thermal_ext, dof=dof)
            replace_img_names(save_dir, undist_rgb_ext, check=False)

        elif mode == "UNALIGNED":
            # unaligned (identity homography)
            shutil.copytree(thermal_texture_dir, save_dir)
            replace_img_names(save_dir, undist_rgb_ext, check=False)

        if optimize_disk_space:
            shutil.rmtree(f"{project}/thermal")
            shutil.rmtree(f"{project}/combined/opensfm/undistorted/images_rgb")


    ## STAGE 4 -- THERMAL ORTHOMOSAIC TEXTURING
    if cfg["STAGES"]['COMBINED_STAGES']['STAGE_4']:
        print("\n\n------------ STAGE 4 ------------")
        mode = cfg["HOMOGRAPHY"]["MODE"]

        ## run ODM on thermal data from texturing step
        os.chdir("ODM")
        flags_dir = cfg["ODM"]
        flags_dir.pop('rerun-all')
        flags_dir["rerun-from mvs_texturing"] = ''
        if optimize_disk_space:
            flags_dir["optimize-disk-space"] = ''
        flags = get_ODM_flags(flags_dir)
        second_part = f"{ROOT_DIR}/{project}/combined" if ROOT_DIR not in project else f"{project}/combined"
        call_2 = f".\\run.bat {flags} {second_part}"
        os.system(call_2)
        os.chdir("..")


        ## move orthophoto
        output_path = f'{output_dir}/orthophoto_combined_thermal.tif'
        shutil.move(f'{project}/combined/odm_orthophoto/odm_orthophoto.tif', output_path) 




def thermal_mapping_pipeline(cfg):
    """
    Thermal mapping pipeline.
    Does everything from moving rjpegs to preparing ortho-ready images.
    """
    # set up directories and extract info
    project, mapping_dir, nested, rgb_dir, thermal_dir, output_dir = get_directories(cfg)
    camera = cfg["CAMERA"]
    two_step = cfg["STAGES"]["THERMAL"]["TWO_STEP"]
    normalize = cfg["PREPROCESSING"]["THERMAL"]["NORMALIZE"]
    rgb_ext, _ = determine_extensions(camera, rgb_dir, thermal_dir)
    optimize_disk_space = cfg["OUTPUT"]["OPTIMIZE_DISK_SPACE"]
    ROOT_DIR = os.path.abspath(os.getcwd()).replace('\\', '/')
    try: shutil.rmtree(f'{project}/thermal_only')
    except: pass

    if camera == "H20T":
        # Move RJPEGs to one directory
        print("Moving RJPEGs to single directory...")
        if mapping_dir is not None:
            if nested:
                inner_dirs = os.listdir(f"{mapping_dir}")
                for dir in inner_dirs:
                    prepare_thermal_images(f'{mapping_dir}/{dir}', f'{project}/thermal_only/rjpegs', ext="_T.JPG")
            else:
                prepare_thermal_images(f'{mapping_dir}', f'{project}/thermal_only/rjpegs', ext="_T.JPG")
        else:
            prepare_thermal_images(f'{thermal_dir}', f'{project}/thermal_only/rjpegs', ext="")
        
        # Convert RJPEGs to raw binary files / temp files
        print("Converting rjpegs to raw...")
        function = cfg["PREPROCESSING"]["THERMAL"]["H20T"]["FUNCTION"]
        exec_path = cfg["PREPROCESSING"]["THERMAL"]["H20T"]["DJI_THERMAL_SDK_PATH"]
        if exec_path is None or (not os.path.exists(exec_path)):
            exec_path = f"{ROOT_DIR}/DJI_Thermal_SDK/utility/bin/windows/release_x64/dji_irp_omp.exe"
        convert_rjpeg_to_raw(exec_path, f'{project}/thermal_only/rjpegs', f'{project}/thermal_only', f"{project}/logs/log_binary-extraction.txt", function=function)

        ## rasterize binary data to TIFFs (output dir dpepends on perprocessing options)
        print("Converting raw to tiffs...")
        if normalize or two_step:
            rasterize_raw_to_tiffs(f'{project}/thermal_only/raw', f'{project}/thermal_only/tiffs', ROOT_DIR, function)                 
        else:
            rasterize_raw_to_tiffs(f'{project}/thermal_only/raw', f'{project}/thermal_only/images', ROOT_DIR, function) 

        ## normalize
        if two_step:
            print("Hist Equalization for all images to increase contrast...")
            local_hist_eq(f'{project}/thermal_only/tiffs', f"{project}/thermal_only/images")
            if normalize:
                print("Normalizing...")
                min_max_norm_images(f'{project}/thermal_only/tiffs', f"{project}/thermal_only/normalized", f"{project}/logs/log_normalization.txt")
        elif normalize:
            print("Normalizing...")
            min_max_norm_images(f'{project}/thermal_only/tiffs', f"{project}/thermal_only/images", f"{project}/logs/log_normalization.txt")

        ## copy EXIF data from rjpeg to tiffs 
        print("Inserting EXIF data...")
        insert_thermal_tiff_exifs(f'{project}/thermal_only/rjpegs', f'{project}/thermal_only/images', f"{project}/logs/log_exiftool.txt")
        

    # for other drones
    else:
        # determine and do preprocessing
        if two_step or normalize:
            if two_step:
                print("Hist Equalization for all images to increase contrast...")
                local_hist_eq(thermal_dir, f'{project}/thermal_only/images', ext="")
                if normalize:
                    min_max_norm_images(thermal_dir, f"{project}/thermal_only/normalized", f"{project}/logs/log_normalization.txt")
            else:
                min_max_norm_images(thermal_dir, f"{project}/thermal_only/images", f"{project}/logs/log_normalization.txt")

            # copy over exif data
            print("Inserting EXIF data...")
            insert_thermal_tiff_exifs(thermal_dir, f'{project}/thermal_only/images', f"{project}/logs/log_exiftool.txt")
        else:
            # no preprocessing needed
            prepare_thermal_images(thermal_dir, f'{project}/thermal_only/images', ext="")

    # ODM
    # run ODM on thermal data
    os.chdir("ODM")
    flags = get_ODM_flags(cfg["ODM"])
    second_part = f"{ROOT_DIR}/{project}/thermal_only" if ROOT_DIR not in project else f"{project}/thermal_only"
    call_1 = f".\\run.bat {flags} {second_part}"
    os.system(call_1)
    os.chdir("..")

    # check
    if not os.path.exists(f"{project}/thermal_only/opensfm/undistorted/images"):
        print("ODM Failed. Stopping thermal only processing. Change ODM parameters in config file and try again.")
        return

    # re-texture if two-step:
    if two_step:
        # move undistorted equalized images
        undist_histeq_dir = f"{project}/thermal_only/opensfm/undistorted/images_histeq"
        shutil.move(f"{project}/thermal_only/opensfm/undistorted/images", undist_histeq_dir)
        undist_histeq_ext = ".".join(os.listdir(undist_histeq_dir)[1].split('.')[1:])

        thermal_final_dir = f"{project}/thermal_only/normalized" if normalize else f'{project}/thermal_only/tiffs'
        thermal_final_dir = thermal_final_dir if camera == 'H20T' else thermal_dir

        # do registration - determine mode first   
        mode = cfg["HOMOGRAPHY"]["MODE"]
        save_dir = f"{project}/thermal_only/opensfm/undistorted/images"
        try: shutil.rmtree(save_dir)
        except: pass
        print("Homography calculation...")
        dof = cfg["HOMOGRAPHY"]["DOF"]
        if mode == "MANUAL":
            ## Homography computation from manual point correspondences
            img_index = cfg["HOMOGRAPHY"]["MANUAL_PARAMS"]["IMAGE_PAIR_INDEX"]
            hist_eq = cfg["HOMOGRAPHY"]["MANUAL_PARAMS"]["HIST_EQ"]
            map_all_images_manually(project, undist_histeq_dir, thermal_final_dir, save_dir, img_index, eq_hist=hist_eq, dof=dof)
            replace_img_names(save_dir, undist_histeq_ext, check=False)

        elif mode =="NGF":
            ## NGF to find homography
            NGF_params = cfg["HOMOGRAPHY"]["NGF_PARAMS"]
            map_all_images_using_NGF(undist_histeq_dir, thermal_final_dir, save_dir, project, NGF_params, dof=dof)
            replace_img_names(save_dir, undist_histeq_ext, check=False)

        elif mode == "UNALIGNED":
            # unaligned (identity homography)
            shutil.copytree(thermal_final_dir, save_dir)
            replace_img_names(save_dir, undist_histeq_ext, check=False)

        os.chdir("ODM")
        flags_dir = cfg["ODM"]
        flags_dir["rerun-from mvs_texturing"] = ''
        flags_dir.pop('rerun-all')
        if optimize_disk_space:
            flags_dir['optimize-disk-space'] = ''
        flags = get_ODM_flags(flags_dir)
        second_part = f"{ROOT_DIR}/{project}/thermal_only" if ROOT_DIR not in project else f"{project}/thermal_only"
        call_2 = f".\\run.bat {flags} {second_part}"
        os.system(call_2)
        os.chdir("..")
    




def rgb_only_pipeline(cfg):
    # set up directories and extract info
    project, mapping_dir, nested, rgb_dir, thermal_dir, output_dir = get_directories(cfg)
    camera = cfg["CAMERA"]
    rgb_ext, _ = determine_extensions(camera, rgb_dir, thermal_dir)
    optimize_disk_space = cfg["OUTPUT"]["OPTIMIZE_DISK_SPACE"]
    ROOT_DIR = os.path.abspath(os.getcwd()).replace('\\','/')

    # extract options
    crop = cfg['PREPROCESSING']['RGB']['CROP']
    scale = cfg['PREPROCESSING']['RGB']['SCALE']

    # remove img dir:
    try: shutil.rmtree(f'{project}/rgb_only/images')
    except: pass
    
    # move all RGB images to single directory (also crop and copy over EXIFs properly)
    if camera == "H20T" and mapping_dir is not None:
        if nested:
            inner_dirs = os.listdir(mapping_dir)
            for dir in inner_dirs:
                prepare_mosaic_images(f'{mapping_dir}/{dir}', f'{project}/rgb_only/images', crop=crop, scale=scale, overwrite=True, ext="_W.JPG")
        else:
            prepare_mosaic_images(f'{mapping_dir}', f'{project}/rgb_only/images', crop=crop, scale=scale, overwrite=True, ext="_W.JPG")
    else:
        prepare_mosaic_images(f'{rgb_dir}', f'{project}/rgb_only/images', crop=crop, scale=scale, overwrite=True, ext="")
    replace_img_names(f'{project}/rgb_only/images', ext=rgb_ext)

    # run ODM on RGB data
    os.chdir("ODM")
    flags_dict = cfg["ODM"]    
    if optimize_disk_space:
        flags_dict['optimize-disk-space'] = ''
    flags = get_ODM_flags(flags_dict)
    second_part = f"{ROOT_DIR}/{project}/rgb_only" if ROOT_DIR not in project else f"{project}/rgb_only"
    call_1 = f".\\run.bat {flags} {second_part}"
    os.system(call_1)
    os.chdir("..")

    # Move output
    print("Moving RGB ortho...")
    shutil.move(f'{project}/rgb_only/odm_orthophoto/odm_orthophoto.tif', f'{output_dir}/orthophoto_rgb_only.tif') 





### --------- DOWNSTREAM TASKS -----------

def tree_detection(cfg):
    # determine which ortho to use
    method = cfg["STAGES"]["METHOD"]
    project = cfg["DIR"]["PROJECT"] 
    denorm = cfg["OUTPUT"]["DENORMALIZE"]

    # make output dir for tree detection
    output_dir = f'{project}/output/tree_detection'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # get model
    model = main.deepforest()
    model.use_release()

    # extract params
    params = cfg["STAGES"]["TREE_DETECTION"]
    thresh = params["SCORE_THRESH"]
    patch_size = params["PATCH_SIZE"]
    patch_overlap = params["PATCH_OVERLAP"]
    save_image = params["SAVE_BBOX_IMAGE"]
    save_crowns = params["SAVE_TREE_CROWNS"]
    iou_threshold = params['IOU_THRESH']
    n_top = params["N_TOP"]

    if method == 'combined':
        denorm_flag = '_non-norm' if denorm else ''
        mode =  cfg['HOMOGRAPHY']['MODE']
        rgb_ortho_path =  f'{project}/output/orthophoto_combined_rgb.tif'
        thermal_ortho_path = f'{project}/output/orthophoto_combined_thermal_{mode}{denorm_flag}.tif'

        if not (os.path.exists(rgb_ortho_path) and os.path.exists(thermal_ortho_path)):
            print(f"Uh oh! It seems one or both of the orthos were not generated. Check and try again")
            return
        
        # load and convert types
        rgb_ortho = cv2.cvtColor(cv2.imread(rgb_ortho_path), cv2.COLOR_BGR2RGB)
        thermal_ortho = cv2.resize(cv2.imread(thermal_ortho_path, cv2.IMREAD_UNCHANGED),(rgb_ortho.shape[1], rgb_ortho.shape[0]), cv2.INTER_CUBIC)

        # run model
        predicted_raster = model.predict_tile(image=rgb_ortho, return_plot = False, patch_size=patch_size,patch_overlap=patch_overlap, color=(255,0,0), thresh=thresh, iou_threshold=iou_threshold)
        print(f"Predicted {len(predicted_raster)} tree crowns.")

        # save image
        if save_image:
            predicted_image = model.predict_tile(image=rgb_ortho, return_plot = True, patch_size=patch_size,patch_overlap=patch_overlap, color=(255,0,0), thresh=thresh, iou_threshold=iou_threshold)
            predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGBA)
            cv2.imwrite(f"{output_dir}/predicted_image.jpeg", predicted_image)


        # plot best predictions         
        r = int(np.ceil(np.sqrt(n_top)))

        i = 0
        fig, axes = plt.subplots(r,r)
        fig.set_figheight(8)
        fig.set_figwidth(16)
        gs0 = gridspec.GridSpec(1, 2, figure=fig)
        gs00 = gridspec.GridSpecFromSubplotSpec(r, r, subplot_spec=gs0[0])
        gs01 = gs0[1].subgridspec(r, r)
        plt.suptitle(f"Subset of Predicted Trees from {method} Orthomosaic")
        

        for idx in range(len(predicted_raster)):
            shift = False
            if idx in [0,1,2,13]:
                shift = True
                # continue
            if idx in [1,2,9,11,12]:
                continue
            if i >= n_top:
                break

            # check if valid
            e = predicted_raster.iloc[idx]
            xmin, ymin, xmax, ymax = int(e['xmin']), int(e['ymin']), int(e['xmax']), int(e['ymax'])
            rgb_crown = rgb_ortho[ymin:ymax, xmin:xmax] 
            thermal_crown = thermal_ortho[ymin+4:ymax+4, xmin-8:xmax-8] if not shift else thermal_ortho[ymin+1:ymax+1, xmin-8:xmax-8]
            if np.sum(rgb_crown) == 0 or np.sum(thermal_crown) == 0:
                continue

            # titles
            plt.axis('off')
            ax_rgb = fig.add_subplot(gs00[i//r, i%r])
            ax_thermal = fig.add_subplot(gs01[i//r, i%r])
            ax_rgb.set_title(f"{idx}, {e['score']*100:.2f}%", fontsize=8)
            ax_rgb.imshow(rgb_crown)
            ax_rgb.axis('off')
            ax_thermal.set_title(f"{idx}, {e['score']*100:.2f}%", fontsize=8)
            ax_thermal.imshow(thermal_crown, cmap='gray')
            ax_thermal.axis('off')
            axes[i//r, i%r].axis('off')
            i += 1

        plt.savefig(f'{output_dir}/sample_crowns_combined.pdf')
        plt.axis('off')
        plt.show()

        # save all predictions
        if save_crowns:
            rgb_crowns_dir = f'{output_dir}/rgb_crowns'
            thermal_crowns_dir = f'{output_dir}/thermal_crowns'
            if not os.path.exists(rgb_crowns_dir):
                os.makedirs(rgb_crowns_dir)
            if not os.path.exists(thermal_crowns_dir):
                os.makedirs(thermal_crowns_dir)

            i = 0
            print("Saving valid predicted tree crowns...")
            for idx in tqdm(range(len(predicted_raster))):
                # check if valid
                e = predicted_raster.iloc[idx]
                xmin, ymin, xmax, ymax = int(e['xmin']), int(e['ymin']), int(e['xmax']), int(e['ymax'])
                rgb_crown = rgb_ortho[ymin:ymax, xmin:xmax]  
                thermal_crown = thermal_ortho[ymin:ymax, xmin:xmax] 
                if np.sum(rgb_crown) == 0 or np.sum(thermal_crown) == 0:
                    continue

                # save
                rgb_save_path = f'{rgb_crowns_dir}/crown_{str(i).zfill(6)}.tif'
                thermal_save_path = f'{thermal_crowns_dir}/crown_{str(i).zfill(6)}.tif'
                cv2.imwrite(rgb_save_path, rgb_crown[:,:,::-1])
                cv2.imwrite(thermal_save_path, thermal_crown)
                i += 1


    else:
        # load ortho
        ortho_path = f'{project}/output/orthophoto_{method}.tif'
        if not os.path.exists(ortho_path):
            print(f"Uh oh! The {method} ortho was not generated.")
            return
        ortho = cv2.imread(ortho_path, cv2.IMREAD_UNCHANGED)
        
        # get alpha channel and convert to RGB if needed
        if method == 'rgb_only':
            ortho = cv2.cvtColor(ortho, cv2.COLOR_BGR2RGB) 
        else:
            ortho = cv2.cvtColor(ortho, cv2.COLOR_GRAY2RGB)
            ortho_unchanged = cv2.imread(ortho_path, cv2.IMREAD_UNCHANGED)        

        # run model
        predicted_raster = model.predict_tile(image=ortho, return_plot = False, patch_size=patch_size,patch_overlap=patch_overlap, color=(255,0,0), thresh=thresh, iou_threshold=iou_threshold)
        print(f"Predicted {len(predicted_raster)} tree crowns.")

        # save image
        if save_image:
            predicted_image = model.predict_tile(image=ortho, return_plot = True, patch_size=patch_size,patch_overlap=patch_overlap, color=(255,0,0), thresh=thresh, iou_threshold=iou_threshold)
            predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGBA)
            cv2.imwrite(f"{output_dir}/predicted_image.jpeg", predicted_image)


        # plot best predictions
        r = int(np.ceil(np.sqrt(n_top)))

        i = 0
        fig, axes = plt.subplots(r,r)
        fig.set_figheight(10)
        fig.set_figwidth(10)
        plt.suptitle(f"Subset of Predicted Trees from {method} Orthomosaic")
        for idx in range(len(predicted_raster)):
            # if idx in [7,11,37, 40, 46, 48]:
            #     continue
            if i >= n_top:
                break

            # check if valid
            e = predicted_raster.iloc[idx]
            xmin, ymin, xmax, ymax = int(e['xmin']), int(e['ymin']), int(e['xmax']), int(e['ymax'])
            crown = ortho[ymin:ymax, xmin:xmax] 
            print(crown.shape)
            if np.sum(crown) == 0:
                continue

            # titles
            axes[i//r, i%r].set_title(f"{idx}, {e['score']*100:.2f}%", fontsize=8)
            axes[i//r, i%r].imshow(crown)
            axes[i//r, i%r].axis('off')
            i += 1

        plt.savefig(f'{output_dir}/sample_crowns.jpeg')
        plt.show()

        # save all predictions
        if save_crowns:
            crowns_dir = f'{output_dir}/crowns'
            if not os.path.exists(crowns_dir):
                os.makedirs(crowns_dir)

            i = 0
            print("Saving valid predicted tree crowns...")
            for idx in tqdm(range(len(predicted_raster))):

                # check if valid
                e = predicted_raster.iloc[idx]
                xmin, ymin, xmax, ymax = int(e['xmin']), int(e['ymin']), int(e['xmax']), int(e['ymax'])
                crown = ortho[ymin:ymax, xmin:xmax]  if method == 'rgb_only' else ortho_unchanged[ymin:ymax, xmin:xmax] 
                if np.sum(crown) == 0:
                    continue

                # save
                save_path = f'{crowns_dir}/crown_{str(i).zfill(6)}.tif'
                if method == 'rgb_only':
                    cv2.imwrite(save_path, crown[:,:,::-1])
                else:
                    cv2.imwrite(save_path, crown)
                i += 1



            
            

        




import pprint
def main_fn():
    cfg_path = sys.argv[1]
    if not os.path.exists(cfg_path):
        print("Please specify a valid config path.")
        return

    with open(cfg_path, 'rb') as f:
        cfg = yaml.safe_load(f.read())


    # find root dir and stuff
    project_path = cfg["DIR"]["PROJECT"] 
    if not os.path.exists(project_path):
        print(f"Please specify a valid project path in {cfg_path}.")
        return

    # determine which mosaics to generate
    method = cfg['STAGES']['METHOD']
    if method == 'combined':  
        combined_mapping_pipeline(cfg)

    # rgb only
    elif method == 'rgb_only':
        rgb_only_pipeline(cfg)

    # thermal only
    elif method == 'thermal_only':
        thermal_mapping_pipeline(cfg)


    # Downstream tasks:
    if cfg["STAGES"]["TREE_DETECTION"]["PERFORM"]:
        tree_detection(cfg)




    




if __name__ == "__main__":
    main_fn() 


