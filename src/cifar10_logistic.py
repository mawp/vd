import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    '''
    Returns a 10000, 3072 array which means 10000 32x32 images in r, g, b.
    '''
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding = 'latin1')
    fo.close()
    return dict

def load_all_data():
    train = [
        unpickle('/home/bdi/data/cifar-10-batches-py/data_batch_1'),
        unpickle('/home/bdi/data/cifar-10-batches-py/data_batch_2'),
        unpickle('/home/bdi/data/cifar-10-batches-py/data_batch_3'),
        unpickle('/home/bdi/data/cifar-10-batches-py/data_batch_4'),
        unpickle('/home/bdi/data/cifar-10-batches-py/data_batch_5'),
    ]
    test = unpickle('/home/bdi/data/cifar-10-batches-py/test_batch')
    train_d = np.vstack([x['data'] for x in train])
    train_l = np.hstack([x['labels'] for x in train])
    test_d = np.array(test['data'])
    test_l = np.array(test['labels'])
    meta  = unpickle('/home/bdi/data/cifar-10-batches-py/batches.meta')
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

def softmax(train_d, train_l, test_d, test_l):
    '''
    '''
    x = tf.placeholder(tf.float32, [None, 32*32*3])
    W = tf.Variable(tf.zeros([32*32*3, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])
    )
    train_step = tf.train.GradientDescentOptimizer(0.5).\
                 minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = train_d[i*50:(i+1)*50],\
                             train_l[i*50:(i+1)*50]
        fd = {x:batch_xs, y_: batch_ys}
        train_accuracy = accuracy.eval(session=sess, feed_dict=fd)
        print("step %d, training accuracy %g"%(i, train_accuracy))
        sess.run(train_step, feed_dict=fd)

    print(sess.run(accuracy, feed_dict={x: test_d, y_: test_l}))

def main():

    train_d, train_l, train_l_onehot, test_d, test_l, test_l_onehot, classes \
        = load_all_data()
    show_rgb(train_d,train_l,classes,3907)
    #softmax(train_d, train_l_onehot, test_d, test_l_onehot)

if __name__ == '__main__':
    main()
