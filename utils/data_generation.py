import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import listdir
from os.path import isfile, join
import json
from skimage import draw
import numpy as np
from PIL import Image
import PIL
from scipy import ndimage

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=int)
    mask[fill_row_coords, fill_col_coords] = 255 # white
    return mask

def show_orig_img(filename, dir_path):
    img = mpimg.imread(dir_path + filename)
    return img, img.shape

def generate_mask_images(filepath, dir_path, mask_dir_path):
    f = open(filepath,)
    data = json.load(f)

    for i in data:
        file = data[i]['filename']
        region = data[i]['regions'][0]['shape_attributes']
        all_points_x = region['all_points_y']
        all_points_y = region['all_points_x']
        (orig, dim) = show_orig_img(file, dir_path)
        mask = poly2mask(all_points_x, all_points_y, (dim[0], dim[1]))

        img = mask.astype('uint8') * 255
        
        cv2.imwrite(mask_dir_path + file.split(".")[0] + "_mask.jpg", mask)

#         im = Image.open(mask_dir_path + file.split(".")[0] + "_mask.jpg")
#         im.save(mask_dir_path + file.split(".")[0] + "_mask.gif") # Save img with 4 channels
        
        show_img_with_mash(orig, img)

    f.close()

def print_img_in_folder(path, show):
    imgs = [f for f in listdir(path) if isfile(join(path, f))]
    if show:
        print(imgs)
    return imgs

def show_img_with_mash(img, mask):
    rows=1
    cols = 2
    img_count = 0

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(28,20))
    axes[0].imshow(img)
    axes[1].imshow(mask)
