#################################
# Your name: noa erez
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt
from scipy.special import softmax
from numpy import linalg


"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels



def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    data_len = data.shape[0]
    data_dim = data.shape[1]

    w = np.zeros(data_dim)

    for i in range(1, T+1):
        sample = np.random.randint(0, data_len, 1)[0]
        y_i = labels[sample]
        x_i = data[sample]
        eta_t = eta_0 / i

        if y_i * np.dot(w,x_i) < 1:
            w = (1-eta_t) * w + (eta_t * C * y_i * x_i)
        else:
            w = (1-eta_t) * w
    return w



def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """

    def Log_loss_gradient_calculation(data, labels, i, w):
        x_i = data[i]
        y_i = labels[i]
        exp = [-y_i * np.dot(w, x_i), 0]
        grad = (softmax(exp)[0] * -1 * y_i) * x_i
        return grad

    data_len = data.shape[0]
    data_dim = data.shape[1]

    w = np.zeros(data_dim)

    for i in range(1, T + 1):
        sample = np.random.randint(0, data_len, 1)[0]
        eta_t = eta_0 / i
        w = w - (eta_t * Log_loss_gradient_calculation(data, labels, sample, w))

    return w


#################################

# Place for additional code

# Q1

# a
def average_accuracy_on_validation_set_as_func_of_eta_0 (train_data, train_labels, validation_data, validation_labels, C, T):

    etaArr = [10 ** i for i in range(-5, 6)]
    AvgAccuracyArr = []

    for eta in etaArr:
        accuracySum = 0
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, C, eta, T)
            acc = accuracy_calculation(validation_data, validation_labels, w)
            accuracySum += acc

        AvgAccuracyArr.append( accuracySum/10 )


    plt.plot(etaArr, AvgAccuracyArr)
    plt.title("Average accuracy on the validation set, as a function of eta_0")
    plt.ylabel("Average accuracy")
    plt.xlabel("eta_0")
    plt.xscale('log') # ?
    plt.show()

    # return the best eta
    return etaArr[np.argmax(AvgAccuracyArr)]



def accuracy_calculation (data, labels, w):
    data_len = data.shape[0]
    correct_prediction_count = 0

    for i in range(data_len):
        x_i = data[i]
        y_i = labels[i]

        predict = 1 if np.dot(w,x_i) >= 0 else -1
        correct_prediction_count += 1 if predict == y_i else 0

    return correct_prediction_count / data_len

# b
def average_accuracy_on_validation_set_as_func_of_C (train_data, train_labels, validation_data, validation_labels, T, eta_0):

    CArr = [10 ** i for i in range(-5, 6)]
    accuracySum = 0
    AvgAccuracyArr = []

    for C in CArr:
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, C, eta_0, T)
            acc = accuracy_calculation(validation_data, validation_labels, w)
            accuracySum += acc
        AvgAccuracyArr.append(accuracySum / 10)
        accuracySum = 0

    plt.plot(CArr, np.array(AvgAccuracyArr))
    plt.title("Average accuracy on the validation set, as a function of C")
    plt.ylabel("Average accuracy")
    plt.xlabel("C")
    plt.xscale('log')  # ?
    plt.show()

    # return the best C given the best eta_0
    return CArr[np.argmax(AvgAccuracyArr)]

# c
def Q1_section_c (train_data, train_labels, best_C, best_eta_0, T):
    w = SGD_hinge(train_data, train_labels, best_C, best_eta_0, T)
    plt.imshow(np.reshape(w, (28, 28)), interpolation = 'nearest')
    plt.show()

# d
def best_classifier_accuracy_on_test_set (train_data, train_labels, test_data, test_labels, best_C, best_eta_0, T):
    w = SGD_hinge(train_data, train_labels, best_C, best_eta_0, T)
    return accuracy_calculation(test_data, test_labels, w)


# Q2

# a
def average_accuracy_on_validation_set_as_func_of_eta_0 (train_data, train_labels, validation_data, validation_labels, T):

    etaArr = [10 ** i for i in range(-5, 6)]
    AvgAccuracyArr = []

    for eta in etaArr:
        accuracySum = 0
        for i in range(10):
            w = SGD_log(train_data, train_labels, eta, T)
            acc = accuracy_calculation(validation_data, validation_labels, w)
            accuracySum += acc

        AvgAccuracyArr.append( accuracySum/10 )

    plt.plot(etaArr, AvgAccuracyArr)
    plt.title("Average accuracy on the validation set, as a function of eta_0")
    plt.ylabel("Average accuracy")
    plt.xlabel("eta_0")
    plt.xscale('log') # ?
    plt.show()

    # return the best eta
    return etaArr[np.argmax(AvgAccuracyArr)]


# b
def Q2_section_b (train_data, train_labels, best_eta_0, T):
    w = SGD_log(train_data, train_labels, best_eta_0, T)
    plt.imshow(np.reshape(w, (28, 28)), interpolation = 'nearest')
    plt.show()

def best_classifier_accuracy_on_test_set (train_data, train_labels, test_data, test_labels, best_eta_0, T):
    w = SGD_log(train_data, train_labels, best_eta_0, T)
    return accuracy_calculation(test_data, test_labels, w)


# c
def W_norm_as_function_of_iteration(data, labels, eta_0, T):

    def Log_loss_gradient_calculation(data, labels, i, w):
        x_i = data[i]
        y_i = labels[i]
        exp = [-y_i * np.dot(w, x_i), 0]
        grad = (softmax(exp)[0] * -1 * y_i) * x_i
        return grad

    data_len = data.shape[0]
    data_dim = data.shape[1]

    w = np.zeros(data_dim)
    normArr = []

    for i in range(1, T + 1):
        sample = np.random.randint(0, data_len, 1)[0]
        eta_t = eta_0 / i
        w = w - (eta_t * Log_loss_gradient_calculation(data, labels, sample, w))
        normArr.append(linalg.norm(w))

    plt.plot(np.arange(T), np.array(normArr))
    plt.title("Norm of w as a function of the iteration")
    plt.ylabel("Norm of w")
    plt.xlabel("iteration")
    plt.show()





train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()

# Q1

C = 1
T = 1000
best_eta_0 = average_accuracy_on_validation_set_as_func_of_eta_0 (train_data, train_labels, validation_data, validation_labels, C, T)
print("question 1 Section a :\n best eta_0: ", best_eta_0)

best_C_given_eta_0 = average_accuracy_on_validation_set_as_func_of_C (train_data, train_labels, validation_data, validation_labels, T, best_eta_0)
print("question 1 Section b :\n best C: ", best_C_given_eta_0)


T = 20000
Q1_section_c(train_data, train_labels, best_C_given_eta_0, best_eta_0, T)

best_classifier_acc = best_classifier_accuracy_on_test_set (train_data, train_labels, test_data, test_labels, best_C_given_eta_0, best_eta_0, T)
print("best classifier accuracy on test set : ", best_classifier_acc)


# Q2

T = 1000
best_eta_0 = average_accuracy_on_validation_set_as_func_of_eta_0 (train_data, train_labels, validation_data, validation_labels, T)
print("question 1 Section a :\n best eta_0: ", best_eta_0)

T = 20000

Q2_section_b(train_data, train_labels, best_eta_0, T)

best_classifier_acc = best_classifier_accuracy_on_test_set (train_data, train_labels, test_data, test_labels, best_eta_0, T)
print("best classifier accuracy on test set : ", best_classifier_acc)

W_norm_as_function_of_iteration (train_data, train_labels, best_eta_0, T)


#################################