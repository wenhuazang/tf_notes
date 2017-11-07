import gzip
import struct
import cPickle as pkl
import cPickle
import tables
import numpy as np
import os
from sklearn.datasets import load_svmlight_files


def load_amazon(input_name, data_folder=None, suffix='_train.svmlight'):

    if data_folder is None:
        data_folder = 'data/'
    data_file = os.path.join(data_folder, input_name + suffix)
    print 'load data file {}'.format(data_file)
    xs, ys = load_svmlight_files([data_file])
    # Convert sparse matrices to numpy 2D array
    xs = np.array(xs.todense())
    # Convert {-1,1} labels to {0,1} labels
    ys = np.array((ys + 1) / 2, dtype=int)
    return xs, ys

def loadImageSet(filename):
    print "load image set", filename
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0)
    print "head,", head

    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    # [60000]*28*28
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B'  # like '>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset)

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, 1, width * height])
    print "load imgs finished"
    return imgs


def loadLabelSet(filename):
    print "load label set", filename
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)
    print "head,", head
    imgNum = head[1]

    offset = struct.calcsize('>II')
    numString = '>' + str(imgNum) + "B"
    labels = struct.unpack_from(numString, buffers, offset)
    binfile.close()
    labels = np.reshape(labels, [imgNum, 1])

    print 'load label finished'
    return labels


def load_mnist_raw(data_dir, train):
    if train:
        fd = os.path.join(data_dir, 'train-images-idx3-ubyte')
        fl = os.path.join(data_dir, 'train-labels-idx1-ubyte')
    else:
        fd = os.path.join(data_dir, 't10k-images-idx3-ubyte')
        fl = os.path.join(data_dir, 't10k-labels-idx1-ubyte')
    images = loadImageSet(fd)  # [-1, 1, 784]
    labels = loadLabelSet(fl)
    return images, labels


def load_usps(filename, split):
    filename = os.path.join(filename, 'usps_28x28.pkl')
    # filename = os.path.join(root, filename)
    f = gzip.open(filename, 'rb')
    data_set = cPickle.load(f)
    f.close()
    if split:
        images = data_set[0][0]
        labels = data_set[0][1]
    else:
        images = data_set[1][0]
        labels = data_set[1][1]
    images = images.transpose((0, 2, 3, 1)) / 127.5 - 1
    return images, labels


def load_usps_raw(data_dir, split):
    f = tables.open_file(os.path.join(data_dir, 'usps.h5'), mode='r')
    if split is 'train':
        X = f.root.usps.train_X
        y = f.root.usps.train_y
    else:
        X = f.root.usps.test_X
        y = f.root.usps.test_y
    # images = []
    # for img in X:
    #     images.append(np.array(img, dtype=np.float64))
    # return np.array(images), y
    return X, y


def load_mnist_m():
    mnistm = pkl.load(open('/Users/paul/PycharmProjects/mnist_dan/dann/mnistm_data.pkl'))
    mnistm_train = mnistm['train']
    mnistm_test = mnistm['test']
    return mnistm_train, mnistm_test