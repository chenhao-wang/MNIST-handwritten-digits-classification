import numpy as np
from mnist import MNIST

data_path = '../data'
h = 28
w = 28
c = 10
m = 60000
mt = 10000
eps = 1e-7
batch_size = 64
learning_rate = 0.1
num_epochs = 2000
# rate to keep nodes
# dropout_rate = 0.5
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

    # normalize data before feeding
    # train images / labels
    X_train = np.reshape(X_train, (m, h * w))
    X_train, _ = batchnorm_forward(X_train)
    Y_train = np.reshape(Y_train, (m,))

    # test images / labels
    X_test = np.reshape(X_test, (mt, h * w))
    X_test, _ = batchnorm_forward(X_test)
    Y_test = np.reshape(Y_test, (mt,))

    return X_train, Y_train, X_test, Y_test


# randomly shuffle mini_batch
def random_mini_batches(X, Y, mini_batch_size, seed=0):
    N = X.shape[0]
    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(N))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation]

    num_minibatches = int(N / mini_batch_size)
    for k in range(0, num_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: (k + 1) * mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if N % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_minibatches * mini_batch_size:N, :]
        mini_batch_Y = shuffled_Y[num_minibatches * mini_batch_size:N]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    return dout * (cache > 0)


def dropout_forward(x, rate):
    # rate to keep
    mask = (np.random.rand(*x.shape) < rate) / rate
    out = x * mask
    return out, mask


def dropout_backward(dout, mask):
    return dout * mask


def softmax_loss(x, y, w1, w2, rs):
    """
    :param x:Input x
    :param y:labels y
    :return:loss, dx
    """
    # norm with max value and compute probs
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    # sum of corresponding -log(P(yi))
    N = x.shape[0]
    loss = np.sum(-np.log(np.clip(probs[np.arange(N), y], 1e-10, 1.0))) / N
    loss += 0.5 * rs * (np.sum(w1 * w1) + np.sum(w2 * w2))
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    reg_dw1 = rs * w1
    reg_dw2 = rs * w2
    return loss, dx, reg_dw1, reg_dw2


def linear_forward(x, w, b):
    """
    :param x:(N, D)
    :param w: (D, K)
    :param b: (k,)
    :return:
    -out: out = x*w + b
    -cache: (x, w, b)
    """
    # (N,D) * (D,K) => (N,K)

    out = np.dot(x, w) + b
    cache = (x, w, b)

    return out, cache


def linear_backward(dout, cache):
    """
    :param dout:(N,K)
    :param cache:Input x of same shape as dout
    :return:dw, db
    """
    x, w, b = cache
    # (D,N) * (N,K) => (D,K)
    dw = np.dot(x.T, dout)
    # (N,K) => (K,)
    db = np.sum(dout, axis=0)
    dx = np.dot(dout, w.T)
    return dw, db, dx


def batchnorm_forward(x):
    # naive batch norm to 0 mean 1 var

    x_mean = np.mean(x, axis=0)
    x_var = np.var(x, axis=0)
    cache = 1 / np.sqrt(x_var + eps)
    out = (x - x_mean) * cache

    # print(out.shape, cache.shape)
    return out, cache


def batchnorm_backward(dout, cache):
    """
    :param dout:derivative of batch norm layer
    :param cache:sqrt(var+eps)
    :return:dx
    """
    return dout * cache


def initialize_parameters(h0, h1, num_class):
    w1 = np.random.randn(h0, h1)
    b1 = np.zeros((h1,))
    w2 = np.random.randn(h1, num_class)
    b2 = np.zeros((num_class,))

    parameters = save_parameters(w1, b1, w2, b2)

    return parameters


def save_parameters(w1, b1, w2, b2):
    parameters = {"w1": w1,
                  "b1": b1,
                  "w2": w2,
                  "b2": b2}
    return parameters


def predict(parameters, X, Y):
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]

    Z1, cache1 = linear_forward(X, w1, b1)
    A1, _ = relu_forward(Z1)
    bn1, cache_bn1 = batchnorm_forward(A1)
    Z2, cache2 = linear_forward(bn1, w2, b2)

    Y_pred = np.argmax(Z2, axis=1)

    acc = np.mean(Y_pred == Y)

    return acc, Y_pred
