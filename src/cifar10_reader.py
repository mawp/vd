
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def unpickle(file):
    '''
    Returns a 10000, 3072 array which means 10000 32x32 images in r, g, b.
    '''
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding = 'latin1')
    fo.close()
    return dict

def load_all_data(data_root):
    train = [
        unpickle(os.path.join(data_root,'data_batch_1')),
        unpickle(os.path.join(data_root,'data_batch_2')),
        unpickle(os.path.join(data_root,'data_batch_3')),
        unpickle(os.path.join(data_root,'data_batch_4')),
        unpickle(os.path.join(data_root,'data_batch_5')),
    ]
    test = unpickle(os.path.join(data_root,'test_batch'))
    meta  = unpickle(os.path.join(data_root,'batches.meta'))
    train_d = np.vstack([x['data'] for x in train])
    train_l = np.hstack([x['labels'] for x in train])
    test_d = np.array(test['data'])
    test_l = np.array(test['labels'])
    cls_cnt = len(meta['label_names'])

    # Shuffle training data
    perm = np.random.permutation(range(len(train_l)))
    train_d = train_d[perm,:]
    train_l = train_l[perm]

    train_l_onehot = np.zeros((len(train_l), cls_cnt))
    train_l_onehot[range(len(train_l)),train_l] = 1

    test_l_onehot = np.zeros((len(test_l), cls_cnt))
    test_l_onehot[range(len(test_l)),test_l] = 1

    return train_d, train_l, train_l_onehot, test_d, test_l, test_l_onehot, meta['label_names']

def show_rgb(data,labels,classes,n):
    im = get_rgb_image(data[n,:])
    plt.imshow(im)
    plt.title(classes[labels[n]])
    plt.show()

def get_rgb_image(d):
    r = d[:1024].reshape((32,32))
    g = d[1024:2048].reshape((32,32))
    b = d[2048:].reshape((32,32))
    return np.dstack((r,g,b))

def main():

    desc = '''
    Simple program to read and visualize cifar10 dataset.
    May be used from commandline or loaded as module.
    Remember data is shuffled each time its loaded.
    '''

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', '--path', help='Path where unpacked cifar10 data exists',default = '~')
    parser.add_argument('-n', '--number', help='training number to visualize',default = 1000)
    args = vars(parser.parse_args())
    data_root = args['path']
    train_d, train_l, train_l_onehot, test_d, test_l, test_l_onehot, classes \
        = load_all_data(data_root)
    show_rgb(train_d,train_l,classes,int(args['number']))

if __name__ == '__main__':
    main()
