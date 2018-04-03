from utils import load_data, predict, data_path
import numpy as np

X_train, Y_train, X_test, Y_test = load_data(data_path)

weights = np.load('weights/weights.npy')
parameters = weights.item()

if __name__ == '__main__':
    train_acc, _ = predict(parameters, X_train, Y_train)
    test_acc, _ = predict(parameters, X_test, Y_test)
    print("train accuracy = ", train_acc)
    print("test accuracy = ", test_acc)
