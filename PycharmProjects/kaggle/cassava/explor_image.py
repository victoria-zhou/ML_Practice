import tensorflow as tf
import os
import pathlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def explore_pil():

    train_dir = './data/train'
    train_dir = pathlib.Path(train_dir)
    test_dir = './data/test'
    test_dir = pathlib.Path(test_dir)

    print(os.listdir(train_dir))
    train_count = len(list(train_dir.glob('*/*.jpg')))
    print(f'train_count: {train_count}')
    test_count = len(list(test_dir.glob('*/*.jpg')))
    print(f'test_count: {test_count}')

    infilename = './data/train/cbb/train-cbb-0.jpg'
    img = Image.open(infilename)
    im_arr = np.asarray(img)
    print(im_arr)
    print(im_arr.max())
    print(im_arr.min())
    print(im_arr.shape)
    im_arr_re = im_arr.reshape(666, 500*3)
    plt.imshow(im_arr_re)
    plt.show()

def explore_mpimg():

    img_path = './data/train/cbb/train-cbb-0.jpg'
    img = mpimg.imread(img_path)
    print(img)
    print(img.max())
    print(img.min())
    print(img.shape)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    explore_mpimg()