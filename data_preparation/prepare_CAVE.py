
import os
import h5py
import numpy as np
import random
from PIL import Image
import shutil


def is_img_file(filename):
    """
    Check if input file is an image

    :param filename: path of input file
    """
    return any(filename.endswith(extension) for extension in ['.png', '.bmp'])

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
    Preprocess dataset and store patches to a h5 file.

    : param data_path: path to the unzipped dataset
    : param store_path: where to store generated file
    """
    dataset_name = 'CAVE'

    patch_size = 96
    stride = 32

    file_name = dataset_name + '_train.h5'

    print('Reading image path list ...')
    img_list = [os.path.join(data_path, x) for x in os.listdir(data_path)]
    print ("CAVE trainset contains a total of {} files".format(len(img_list)))
    
    print('Preprocessing image ...')
    hsis = []
    rgbs = []
    for img_path in img_list:
        imgs = [os.path.join(img_path, x) for x in os.listdir(img_path)]
        imgs.sort()
        hsi = []
        rgb = None
        for img in imgs:
            if is_img_file(img):
                img_data = Image.open(img)
                data = np.array(img_data)
                # if end with .bmp, it is a RGB image
                if img.endswith('.bmp'):
                    rgb = data
                    continue
                if (len(data.shape)>2):
                    data = np.uint16(data[:, :, 0] / 255. * 65535) # a special case is 8 bit RGBA image
                hsi.append(data)

        assert rgb is not None
        hsi = np.stack(hsi, axis=2)
        hsi_patches = crop_image(hsi, patch_size, stride)
        rgb_patches = crop_image(rgb, patch_size, stride)
        print("Image {} has a total {} patches".format(img_path, hsi_patches.shape[0]))
        hsis.append(hsi_patches)
        rgbs.append(rgb_patches)

    # stack together ans sample
    hsis = np.concatenate(hsis)
    rgbs = np.concatenate(rgbs)
    print('Saving to h5 file ...')
    # save dataset
    h5f = h5py.File(os.path.join(store_path, file_name), 'w')
    h5f.create_dataset('hsi', data=hsis)
    h5f.create_dataset('rgb', data=rgbs)
    h5f.close()
    print('Finish generating h5 data')


def gen_test(data_path, store_path):
    """
    Preprocess CAVE dataset and store patches to a h5 file.

    : param data_path: path to the unzipped dataset
    : param store_path: where to store generated file
    """
    dataset_name = 'CAVE'
    file_name = dataset_name + '_test.h5'

    print('Reading image path list ...')
    img_list = [os.path.join(data_path, x) for x in os.listdir(data_path)]
    print ("CAVE testset contains a total of {} files".format(len(img_list)))
    
    print('Preprocessing image ...')
    hsis = []
    rgbs = []
    for img_path in img_list:
        imgs = [os.path.join(img_path, x) for x in os.listdir(img_path)]
        imgs.sort()
        hsi = []
        rgb = None
        for img in imgs:
            if is_img_file(img):
                img_data = Image.open(img)
                data = np.array(img_data)
                # if end with .bmp, it is a RGB image
                if img.endswith('.bmp'):
                    rgb = data
                    continue
                if (len(data.shape)>2):
                    data = np.uint16(data[:, :, 0] / 255. * 65535) # a special case is 8 bit RGBA image
                hsi.append(data)

        assert rgb is not None
        hsi = np.stack(hsi, axis=2)
        hsis.append(hsi)
        rgbs.append(rgb)

    # stack together ans sample
    hsis = np.stack(hsis)
    rgbs = np.stack(rgbs)
    print('Saving to h5 file ...')
    # save dataset
    h5f = h5py.File(os.path.join(store_path, file_name), 'w')
    h5f.create_dataset('hsi', data=hsis)
    h5f.create_dataset('rgb', data=rgbs)
    h5f.close()
    print('Finish generating h5 data')


def main(src_path, store_path, num_train=20):
    
    img_list = os.listdir(src_path)
    img_list.sort()

    assert num_train < len(img_list)
    random.shuffle(img_list)
    
    train_list = img_list[:num_train]
    test_list = img_list[num_train:]
    
    train_path = os.path.join(store_path, 'train')
    test_path = os.path.join(store_path, 'test')
    
    for img in train_list:
        path1 = os.path.join(os.path.join(src_path, img), img)
        path2 = os.path.join(train_path, img)
        shutil.copytree(path1, path2)
    
    for img in test_list:
        path1 = os.path.join(os.path.join(src_path, img), img)
        path2 = os.path.join(test_path, img)
        shutil.copytree(path1, path2)
    
    gen_train(train_path, store_path)
    gen_test(test_path, store_path)

if __name__ == "__main__":
    data_path = './complete_ms_data/'
    store_path = './CAVE_h5/'
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    main(data_path, store_path)
