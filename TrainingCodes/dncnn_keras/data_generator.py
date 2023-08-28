# -*- coding: utf-8 -*-

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26}, 
#    number={7}, 
#    pages={3142-3155}, 
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/husqin/DnCNN-keras
# =============================================================================

# no need to run this code separately


import glob
#import os
import cv2
import numpy as np
#from multiprocessing import Pool


patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128


def show(x,title=None,cbar=False,figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x,interpolation='nearest',cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

def data_aug(img, mode=0):

    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def gen_patches(file_name):

    # read image
    img = cv2.imread(file_name, 0)  # gray scale
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s),int(w*s)
        img_scaled = cv2.resize(img, (h_scaled,w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img_scaled[i:i+patch_size, j:j+patch_size]
                #patches.append(x)        
                # data aug
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0,8))
                    patches.append(x_aug)
                
    return patches

def datagenerator(data_dir='data/Train400', batch_size=128, verbose=False):
    file_list = glob.glob(data_dir + '/*.png')  # get name list of all .png files
    
    data = []
    
    for i, file_name in enumerate(file_list):
        patches = gen_patches(file_name)
        data.extend(patches)
        
        if verbose:
            print(f"{i+1}/{len(file_list)} is done ^_^")
    
    # Convert the list of patches to a NumPy array
    data = np.array(data, dtype='uint8')
    
    # Calculate how many patches to discard for batch compatibility
    discard_n = len(data) - len(data) // batch_size * batch_size
    data = data[:-discard_n]
    
    # Reshape data if patches have a consistent shape
    data = data.reshape((-1, patch_size, patch_size, 1))
    
    print('^_^-training data finished-^_^')
    return data


if __name__ == '__main__':   

    data = datagenerator(data_dir='data/Train400')
    

#    print('Shape of result = ' + str(res.shape))
#    print('Saving data...')
#    if not os.path.exists(save_dir):
#            os.mkdir(save_dir)
#    np.save(save_dir+'clean_patches.npy', res)
#    print('Done.')       