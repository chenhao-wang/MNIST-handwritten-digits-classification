import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mnist import MNIST

data_path = '../data'
h = 28
w = 28
c = 10
m = 60000
mt = 10000
eps = 1e-7
batch_size = 64
learning_rate = 1e-3
num_epochs = 200
# rate to keep nodes
dropout_rate = 0.5
regularization_strengths = 1e-7


# load and normalize input data
def load_data(path):
    # load data
    mndata = MNIST(path)
    # list data
    X_train, Y_train = mndata.load_training()
    X_test, Y_test = mndata.load_testing()
    m = len(X_train)
    mt = len(X_test)

    # train images / labels
    X_train = np.reshape(X_train, (m, h, w, 1))
    # X_train = np.reshape(X_train, (m, h * w)) / 255.0
    one_hot_labels = np.zeros((m, c))
    one_hot_labels[np.arange(m), Y_train] = 1
    Y_train = one_hot_labels

    # test images / labels
    X_test = np.reshape(X_test, (mt, h, w, 1))
    # X_test = np.reshape(X_test, (mt, h * w)) / 255.0
    ohl = np.zeros((mt, c))
    ohl[np.arange(mt), Y_test] = 1
    Y_test = ohl

    return X_train, Y_train, X_test, Y_test


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.1, dropout_rate=0.5, num_epochs=2000,
          batch_size=64):
    X = tf.placeholder(dtype=tf.float32, shape=(None, h, w, 1))
    # X = tf.placeholder(dtype=tf.float32, shape=(None, h * w))
    Y = tf.placeholder(dtype=tf.float32, shape=(None, c))

    # two layers neural network
    # h1 = tf.layers.dense(X, 200, activation=tf.nn.relu)
    # dropout = tf.layers.dropout(h1, dropout_rate)
    # batchnorm = tf.layers.batch_normalization(dropout)
    # logits = tf.layers.dense(batchnorm, 10)

    # two conv net
    X_norm = tf.layers.batch_normalization(X)
    conv1 = tf.layers.conv2d(X_norm, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, [2, 2], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool2, (-1, 7 * 7 * 64))
    dense = tf.layers.dense(pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(dense, rate=dropout_rate)

    logits = tf.layers.dense(dropout, 10)

    loss = tf.losses.softmax_cross_entropy(Y, logits)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    # evaluate
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Produces a slice of each Tensor in tensor_list(Queue)
    x_queue, y_queue = tf.train.slice_input_producer([X_train, Y_train], shuffle=False, num_epochs=num_epochs)
    x_batches, y_batches = tf.train.shuffle_batch([x_queue, y_queue], batch_size=batch_size,
                                                  capacity=10000 + 3 * batch_size,
                                                  min_after_dequeue=10000,
                                                  allow_smaller_final_batch=True)
    # we can also manually use loop to control num_epoch
    # x_batches, y_batches = tf.train.shuffle_batch([X_train, Y_train], batch_size=batch_size,
    #                                               capacity=10000 + 3 * batch_size,
    #                                               min_after_dequeue=10000, enqueue_many=True,
    #                                               allow_smaller_final_batch=True)
    costs = []
    saver = tf.train.Saver()
    N = X_train.shape[0]
    num_batches = int(N / batch_size)
    if N % num_batches != 0:
        num_batches += 1

    with tf.Session() as sess:
        # remember to initialize local variable for created parameters like numpy_epochs
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # for epoch in range(1, num_epochs + 1):
        #     epoch_cost = 0.
        #     for step in range(num_batches):
        #         x_batch, y_batch = sess.run([x_batches, y_batches])
        #         _, batch_cost = sess.run([train_op, loss],
        #                                  feed_dict={X: x_batch, Y: y_batch})
        #         epoch_cost += batch_cost
        #
        #     epoch_cost /= num_batches
        #     if epoch % 5 == 0:
        #         print("epoch_cost %d= " % epoch, epoch_cost)
        #         costs.append(epoch_cost)

        # Second version run with num_epochs control from slice_input_producer
        step = 0
        try:
            while not coord.should_stop():
                x_batch, y_batch = sess.run([x_batches, y_batches])
                # Run training steps or whatever
                _, batch_cost = sess.run([train_op, loss],
                                         feed_dict={X: x_batch, Y: y_batch})
                step += 1
                if step % 500 == 0:
                    print("batch_cost %d= " % step, batch_cost)
                    costs.append(batch_cost)
        except tf.errors.OutOfRangeError:
            # When done, ask the threads to stop.
            print('Done training -- epoch limit reached WCH')
        finally:
            coord.request_stop()

        # Wait for threads to finish.
        coord.request_stop()
        coord.join(threads)
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        saver.save(sess, './model/mymodel.ckpt')

        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_data(data_path)
    model(X_train=X_train,
          Y_train=Y_train,
          X_test=X_test,
          Y_test=Y_test,
          learning_rate=learning_rate,
          dropout_rate=dropout_rate,
          num_epochs=num_epochs,
          batch_size=batch_size)
