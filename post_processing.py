import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import label, generate_binary_structure
from skimage import io, color, measure, img_as_float, img_as_uint, img_as_ubyte, exposure, morphology, feature
from skimage.segmentation import clear_border, felzenszwalb, mark_boundaries
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import hessian, frangi, sobel, meijering, sato, threshold_multiotsu, scharr, try_all_threshold, threshold_otsu
from skimage.util import invert
from os.path import join, exists
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2000000000

# 1. Load image
def load_image(file_name,file_path,len_crop_img_px,overlap_px,df_slicing):
    """
    load image from given path
    --------------------------------
    param: (str) file_name: image name
    param: (str) file_path: file name of image
    param: (int) len_crop_img_px: length of cropped image in px
    param: (int) overlap_px: length of overlap of cropped images
    param: (dataframe) df_slicing: slicing parameters of all bags in px
    return: (2darray) img: loaded full raw uint8 image, (int) n: number of cropped images, (dataframe) df_slic_img: slicing parameter for particular image
    """
    df_slic_img = df_slicing[df_slicing.name==file_name[:-4]]
    img_path = join(file_path,file_name)
    img = io.imread(img_path)
    total_length = df_slic_img.px_right.iloc[0]-df_slic_img.px_left.iloc[0]
    n = int((total_length-overlap_px)/(len_crop_img_px-overlap_px))
    print("image loaded")
    return img, n, df_slic_img 

def crop_image(image,i,len_img_px,overlap_px,df_slicing):
    """
    crop image from given full image
    ----------------------------------
    param: (2darray) image: full uint8 image
    param: (int) i: iterating variable
    param: (int) len_img_px: length of cropped image in px
    param: (int) overlap_px: length of overlap of cropped images    
    param: (dataframe) df_slicing: slicing parameter for particular image
    return: (2darray): cropped uint8 image, (2darray): mask of air bubbles of cropped image
    """
    cropped_img = image[df_slicing.px_top.iloc[0]:df_slicing.px_bottom.iloc[0],i*(len_img_px-overlap_px)+df_slicing.px_left.iloc[0]:(i+1)*(len_img_px-overlap_px)+overlap_px+df_slicing.px_left.iloc[0]]
    mask_bubbles = cropped_img < 30 # Mask threshold
    print("image cropped")
    return cropped_img, mask_bubbles

# 2. Denoising 
def denoise(image):
    """
    non-local means filter for denoising and intensity rescale
    ----------------------------------
    param: (2darray) image: 2D uint8 image
    return: (2darray): denoised and rescaled uint8 image
    """
    img_float = img_as_float(image)
    sigma_est = np.mean(estimate_sigma(img_float,channel_axis=None))
    patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                channel_axis=None) # None for greyscale imag
    img_denoised = denoise_nl_means(img_float, h=0.6*sigma_est, sigma=sigma_est,fast_mode=True, **patch_kw)
    img_denoised_uint = img_as_ubyte(img_denoised)
    p2, p95 = np.percentile(img_denoised_uint, (2, 95))
    rescaled = exposure.rescale_intensity(img_denoised_uint, in_range=(p2, p95))
    print("image denoised and rescaled")
    return rescaled
    
# 3. Otsu thresholding
def threshold(image):
    """
    Otsu thresholding of image
    ----------------------------------
    param: (2darray) image: 2D image
    return: (2darray): binary image 
    """
    thresh = threshold_otsu(image)
    binary = image > 130
    print("binary image")
    return binary

# 4. Hessian edge detection
def hessian_filter(image):
    """
    Hessian filter
    ----------------------------------
    param: (2darray) image: 2D binary image
    return: 
    """
    hessian_img = hessian(image, sigmas=[0.1, 0.15], alpha=0.5, beta=0.5, gamma=0.5, black_ridges=True,mode='reflect', cval=0) 
    print("hessian filter")
    return hessian_img

# 5. Morphological filtering
def cleaning(image,mask):
    """
    """
    image_bubbles = image - mask
    image_bubbles = np.where(image_bubbles < 0, 0, image_bubbles)
    clean_image = morphology.diameter_closing(image_bubbles, diameter_threshold=1000)
    print("cleaned image")
    return clean_image

#6. grain sizes label
def label_grains(clean_img,original_img,file_name,n,pixels_to_um):
    mask = clean_img == 1
    cleared_img = clear_border(mask)
    s = generate_binary_structure(2,1)
    labeled_mask, num_labels = label(cleared_img, structure=s)
    label_color = color.label2rgb(labeled_mask)
    uint = img_as_ubyte(label_color)
    if not os.path.exists(join("Plots/NEEM_Labeled_and_Raw_Images", file_name[:-4])):
        os.makedirs(join("Plots/NEEM_Labeled_and_Raw_Images", file_name[:-4]))
    io.imsave("Plots/NEEM_Labeled_and_Raw_Images/{}/{}_{}.png".format(file_name[:-4],file_name[:-4],n),uint)
    io.imsave("Plots/NEEM_Labeled_and_Raw_Images/{}/{}_{}_raw.png".format(file_name[:-4],file_name[:-4],n),original_img)
    print("color labels")
    #fig, ax = plt.subplots(1,1,figsize=(12,7))
    #ax.imshow(label_color)
    #ax.set_title("Labels color")
    #fig.tight_layout()
    #fig.show()
    clusters = measure.regionprops(labeled_mask, original_img)
    
    fname = file_name.replace(".png", "_{}.csv".format(n))    
    propList = ['area',
            'equivalent_diameter',
            'centroid_x',
            'centroid_y',
            'orientation', 
            'major_axis_length',
            'minor_axis_length',
            'perimeter']  
    
    if not os.path.exists(join("/home/jovyan/work/PICE/ResearchGroups/Stratigraphy/Yannick 2022/Data_csv/grain_properties", file_name[:-4])):
        os.makedirs(join("/home/jovyan/work/PICE/ResearchGroups/Stratigraphy/Yannick 2022/Data_csv/grain_properties", file_name[:-4]))
    output_file = open("/home/jovyan/work/PICE/ResearchGroups/Stratigraphy/Yannick 2022/Data_csv/grain_properties/{}/".format(file_name[:-4])+fname, 'w')
    output_file.write(',' + ",".join(propList) + '\n') #join strings in array by commas, leave first cell blank
        
    for cluster_props in clusters:
        #output cluster properties to the excel file
        output_file.write(str(cluster_props['label']))
        for i,prop in enumerate(propList):
            if(prop == 'area'): 
                to_print = cluster_props[prop]#*pixels_to_um**2   #Convert pixel square to um square
            elif(prop == 'orientation'): 
                to_print = cluster_props[prop]*57.2958  #Convert to degrees from radians
            elif(prop == 'centroid_x'): 
                to_print = cluster_props['centroid'][1]
            elif(prop == 'centroid_y'): 
                to_print = cluster_props['centroid'][0]
            else: 
                to_print = cluster_props[prop]     #Reamining props
            output_file.write(',' + str(to_print))
        output_file.write('\n')
    output_file.close()   #Closes the file, otherwise it would be read only. 
    print("written csv")
    
    
##################################### cropping for last image ###########################

def crop_image_last(image,i,len_img_px,overlap_px,df_slicing):
    """
    crop image from given full image
    ----------------------------------
    param: (2darray) image: full uint8 image
    param: (int) i: iterating variable
    param: (int) len_img_px: length of cropped image in px
    param: (int) overlap_px: length of overlap of cropped images    
    param: (dataframe) df_slicing: slicing parameter for particular image
    return: (2darray): cropped uint8 image, (2darray): mask of air bubbles of cropped image
    """
    cropped_img = image[df_slicing.px_top.iloc[0]:df_slicing.px_bottom.iloc[0],i * (len_img_px - overlap_px) + df_slicing.px_left.iloc[0]:df_slicing.px_right.iloc[0]]
    mask_bubbles = cropped_img < 30 # Mask threshold
    print("image cropped")
    return cropped_img, mask_bubbles


