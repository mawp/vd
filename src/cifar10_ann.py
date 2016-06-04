import tensorflow as tf
import cifar10_reader
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
    data_root = '/home/bdi/data/cifar-10-batches-py'
    train_d, train_l, train_l_onehot, test_d, test_l, test_l_onehot, classes \
        = cifar10_reader.load_all_data(data_root)
    softmax(train_d, train_l_onehot, test_d, test_l_onehot)

if __name__ == '__main__':
    main()
