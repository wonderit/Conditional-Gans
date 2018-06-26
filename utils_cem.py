import os
import numpy as np
import pandas as pd
import scipy
import scipy.misc
import matplotlib.pyplot as plt
import cv2
from keras import backend as K


class CEM(object):

    def __init__(self):

        self.dataname = "CEM"
        self.dims = 20*40
        self.shape = [20, 40, 1]
        # self.image_size = 28
        self.data, self.data_y = self.load_mnist()


    def load_mnist(self):

        DATAPATH = '/Users/kimwonsuk/Desktop/cdal/7.CEM/maxwellfdfd_ai_keras/data_train'
        DATASETS = [
            'result_data500',
            'result_data501',
            'result_data502',
            'result_data503',
            'result_data504',
            'result_data505',
            'result_data506',
            'result_data507',
            'result_data508',
            'result_data509',
            '200x100_rl_0002',
            '200x100_rl_0003',
            '200x100_rl_0004',
            '200x100_rl_0005',
            '200x100_rl_0006',
            '200x100_rl_0007',
            '200x100_rl_0008',
            '200x100_rl_0009',
            '200x100_rl_0010',
            '200x100_rl_0011'
        ]

        x_train = []
        y_train = []

        # load dataset
        for data in DATASETS:
            dataframe = pd.read_csv(DATAPATH + '/' + data + '.csv', delim_whitespace=False, header=None)
            dataset = dataframe.values
            # split into input (X) and output (Y) variables
            fileNames = dataset[:, 0]
            # im = skimage.io.imread('a.tif', plugin='tifffile')
            # X_images = []
            y_train.append(dataset[:, 1:25])
            for file in fileNames:
                image = cv2.imread(DATAPATH + '/' + data + '/' + str(int(file)) + '.tiff', 0)
                image = cv2.resize(image, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)

                # cv2.imshow('Shrink', image)
                #
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                image = np.array(image, dtype=np.uint8)
                image //= 255
                x_train.append(image)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        print(x_train.shape)
        print(y_train.shape)

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], self.shape[2], self.shape[0], self.shape[1])
            y_train = y_train.reshape(y_train.shape[0], self.shape[2], self.shape[0], self.shape[1])
            # X_val = X_val.reshape(X_val.shape[0], channels, img_rows, img_cols)
            # X_test = X_test.reshape(X_test.shape[0], channels, img_rows, img_cols)
            input_shape = (self.shape[2], self.shape[0], self.shape[1])
        else:
            x_train = x_train.reshape(x_train.shape[0], self.shape[0], self.shape[1], self.shape[2])
            y_train = y_train.reshape(-1, y_train.shape[2])
            # X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, channels)
            # X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channels)
            input_shape = (self.shape[0], self.shape[1], self.shape[2])

        print('x shape:', x_train.shape)
        print('y shape:', y_train.shape)
        print(x_train.shape[0], 'train samples')
        return x_train, y_train

    def getNext_batch(self, iter_num=0, batch_size=64):

        ro_num = len(self.data) / batch_size - 1

        if iter_num % ro_num == 0:

            length = len(self.data)
            perm = np.arange(length)
            np.random.shuffle(perm)
            self.data = np.array(self.data)
            self.data = self.data[perm]
            self.data_y = np.array(self.data_y)
            self.data_y = self.data_y[perm]

        return self.data[(iter_num % ro_num) * batch_size: (iter_num% ro_num + 1) * batch_size] \
            , self.data_y[(iter_num % ro_num) * batch_size: (iter_num%ro_num + 1) * batch_size]


def get_image(image_path , is_grayscale = False):
    return np.array(inverse_transform(imread(image_path, is_grayscale)))


def save_images(images , size , image_path):
    return imsave(inverse_transform(images) , size , image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def imsave(images , size , path):
    return scipy.misc.imsave(path , merge(images , size))

def merge(images , size):
    h , w = images.shape[1] , images.shape[2]
    img = np.zeros((h*size[0] , w*size[1] , 3))
    for idx , image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h +h , i*w : i*w+w , :] = image

    return img

def inverse_transform(image):
    return (image + 1.)/2.

def read_image_list(category):
    filenames = []
    print("list file")
    list = os.listdir(category)

    for file in list:
        filenames.append(category + "/" + file)

    print("list file ending!")

    return filenames

##from caffe
def vis_square(visu_path , data , type):
    """Take an array of shape (n, height, width) or (n, height, width , 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))

    padding = (((0, n ** 2 - data.shape[0]) ,
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data , padding, mode='constant' , constant_values=1)  # pad with ones (white)

    # tilethe filters into an im age
    data = data.reshape((n , n) + data.shape[1:]).transpose((0 , 2 , 1 , 3) + tuple(range(4 , data.ndim + 1)))

    data = data.reshape((n * data.shape[1] , n * data.shape[3]) + data.shape[4:])

    plt.imshow(data[:,:,0])
    plt.axis('off')

    if type:
        plt.savefig('./{}/weights.png'.format(visu_path) , format='png')
    else:
        plt.savefig('./{}/activation.png'.format(visu_path) , format='png')


def sample_label():
    num = 64
    label_vector = np.zeros((num, 24), dtype=np.float)
    for i in range(0, num):
        label_vector[i, i/8] = 1.0

    print(label_vector.shape)
    return label_vector


def sample_cem_label():
    DATAPATH = '/Users/kimwonsuk/Desktop/cdal/7.CEM/maxwellfdfd_ai_keras/data_test/200x100_rl_0013.csv'
    y_test = []
    dataframe = pd.read_csv(DATAPATH, delim_whitespace=False, header=None)
    dataset = dataframe.values
    y_test.extend(dataset[:8, 1:25])
    y_test = np.array(y_test)
    print(y_test.shape)

    num = 64
    label_vector = []
    for i in range(0, num):
        label_vector.append(y_test[i//8])

    label_vector = np.array(label_vector)
    print(label_vector.shape)
    print(label_vector)
    return label_vector