
import os
import h5py
import numpy as np
import random
import scipy.io as sio
import shutil


def crop_image(img, win, s):
    """
    Crop a hyperspectral image into patches

    :param img: image
    :param win: crop patch size
    :param s: crop stride
    :return: a 4D numpy array contains img patches
    """
    h, w, c = img.shape

    pat_row_num = list(range(0, h - win, s))
    pat_row_num.append(h - win)
    pat_col_num = list(range(0, w - win, s))
    pat_col_num.append(w - win)

    patches = np.zeros((len(pat_row_num)*len(pat_col_num), win, win, c), dtype='float32')
    num = 0

    for i in pat_row_num:
        for j in pat_col_num:
            up = i
            down = up + win
            left = j
            right = left + win
            patches[num, :, :, :] = img[up:down, left:right, :]
            num += 1

    return patches


def gen_train(data_path, store_path):
    """
    Preprocess hyperspectral dataset and store imgs to a h5 file.

    : param data_path: path to the unzipped dataset
    : param store_path: where to store generated file
    """
    dataset_name = 'Harvard'

    patch_size = 96
    stride = 96

    file_name = dataset_name + '_train.h5'

    print('Reading image path list ...')
    img_list = os.listdir(data_path)
    print ("{} trainset contains a total of {} files".format(dataset_name, len(img_list)))
    
    print('Preprocessing image ...')
    hsis = []
    for img in img_list:
        # if end with .mat, it is HSI
        if img.endswith('.mat'):
            data = sio.loadmat(os.path.join(data_path, img))
            hsi = np.float32(data['ref'])[:960,:960,:]
            hsi_patches = crop_image(hsi, patch_size, stride)
            print("Image {} has a total {} patches".format(img, hsi_patches.shape[0]))
            hsis.append(hsi_patches)

    # stack together
    hsis = np.concatenate(hsis)
    print('Saving to h5 file ...')
    # save dataset
    h5f = h5py.File(os.path.join(store_path, file_name), 'w')
    h5f.create_dataset('hsi', data=hsis)
    h5f.close()
    print('Finish generating h5 data')


def gen_test(data_path, store_path):
    """
    Preprocess Harvard dataset and store imgs to a h5 file.

    : param data_path: path to the unzipped dataset
    : param store_path: where to store generated file
    """
    dataset_name = 'Harvard'
    file_name = dataset_name + '_test.h5'

    print('Reading image path list ...')
    img_list = os.listdir(data_path)
    print ("{} testset contains a total of {} files".format(dataset_name, len(img_list)))
    
    print('Preprocessing image ...')
    hsis = []
    for img in img_list:
        # if end with .mat, it is HSI
        if img.endswith('.mat'):
            data = sio.loadmat(os.path.join(data_path, img))
            hsi = np.float32(data['ref'])[:960,:960,:]
            hsis.append(hsi)

    # stack together
    hsis = np.stack(hsis)
    print('Saving to h5 file ...')
    # save dataset
    h5f = h5py.File(os.path.join(store_path, file_name), 'w')
    h5f.create_dataset('hsi', data=hsis)
    h5f.close()
    print('Finish generating h5 data')


def main(src_path, store_path, num_train=30):
    
    img_list = [name for name in os.listdir(src_path) if name.endswith('.mat')]
    img_list.sort()
    print ("Dataset contains a total of {} files".format(len(img_list)))
    
    assert num_train < len(img_list)
    random.shuffle(img_list)
    
    train_list = img_list[:num_train]
    test_list = img_list[num_train:]
    
    train_path = os.path.join(store_path, 'train')
    test_path = os.path.join(store_path, 'test')
    
    for img in train_list:
        path1 = os.path.join(src_path, img)
        path2 = os.path.join(train_path, img)
        shutil.copy(path1, path2)
    
    for img in test_list:
        path1 = os.path.join(src_path, img)
        path2 = os.path.join(test_path, img)
        shutil.copy(path1, path2)
    
    gen_train(train_path, store_path)
    gen_test(test_path, store_path)

if __name__ == "__main__":
    data_path = './Harvard/'
    store_path = './Harvard_h5/'
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    main(data_path, store_path)