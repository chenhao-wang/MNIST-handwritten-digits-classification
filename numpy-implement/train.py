import numpy as np
import matplotlib.pyplot as plt
from utils import *

# load data
X_train, Y_train, X_test, Y_test = load_data(data_path)


def model(X_train, Y_train, X_test, Y_test, learning_rate=learning_rate, num_epochs=2000, minibatch_size=64,
          dropout_rate=1., rs=regularization_strengths):
    parameters = initialize_parameters(h0=h * w, h1=200, num_class=c)
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    seed = 3
    costs = []
    for epoch in range(num_epochs):
        num_minibatches = int(m / minibatch_size)
        epoch_cost = 0.
        seed += 1
        mini_batches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

        for mini_batch in mini_batches:
            (minibatch_x, minibatch_y) = mini_batch
            Z1, cache1 = linear_forward(minibatch_x, w1, b1)
            A1, relu_cache = relu_forward(Z1)
            bn1, cache_bn1 = batchnorm_forward(A1)
            Z2, cache2 = linear_forward(bn1, w2, b2)
            # dpout, mask = dropout_forward(Z2, dropout_rate)

            minibatch_loss, dZ2, reg_dw1, reg_dw2 = softmax_loss(Z2, minibatch_y, w1, w2, rs)
            epoch_cost += minibatch_loss / num_minibatches

            # dZ2 = dropout_backward(d_dpout, mask)
            dw2, db2, dbn1 = linear_backward(dZ2, cache2)
            dw2 += reg_dw2
            dA1 = batchnorm_backward(dbn1, cache_bn1)

            dZ1 = relu_backward(dA1, relu_cache)
            dw1, db1, _ = linear_backward(dZ1, cache1)
            dw1 += reg_dw1

            w2 -= learning_rate * dw2
            b2 -= learning_rate * db2
            w1 -= learning_rate * dw1
            b1 -= learning_rate * db1

        if epoch % 10 == 0:
            print("Iteration %d : train_loss = " % epoch, epoch_cost)
        if epoch % 10 == 0:
            costs.append(epoch_cost)

    # update parameters
    parameters = save_parameters(w1, b1, w2, b2)
    # accuracy
    train_acc, _ = predict(parameters, X_train, Y_train)
    test_acc, _ = predict(parameters, X_test, Y_test)
    print("train accuracy = ", train_acc)
    print("test accuracy = ", test_acc)

    np.save("./weights/weights.npy", parameters)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations(per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

if __name__ == '__main__':
    model(X_train=X_train,
          Y_train=Y_train,
          X_test=X_test,
          Y_test=Y_test,
          learning_rate=learning_rate,
          num_epochs=num_epochs,
          minibatch_size=batch_size,
          rs=regularization_strengths)
