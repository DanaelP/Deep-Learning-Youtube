import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import accuracy_score
from utilities import *
from tqdm import tqdm

def initialisation(dimensions):

    parametres = {}
    C = len(dimensions)

    for c in range(1, C):
        parametres['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
        parametres['b' + str(c)] = np.random.randn(dimensions[c], 1)

    return parametres

def forward_propagation(X, parametres):

    activations = {'A0' : X}
    C = len(parametres) // 2

    for c in range(1, C + 1):
        Z = parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + parametres['b' + str(c)]
        activations['A' + str(c)] = 1 / (1 + np.exp(-Z))

    return activations

def log_loss(y, A):
    epsilon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))

def back_propagation(y, activations, parametres):

    m = y.shape[1]
    C = len(parametres) // 2

    dZ = activations['A' + str(C)] - y
    gradients = {}

    for c in reversed(range(1, C + 1)):
        gradients['dW' + str(c)] = 1 / m * np.dot(dZ, activations['A' + str(c - 1)].T)
        gradients['db' + str(c)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        if c > 1:
            dZ = np.dot(parametres['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])
    
    return gradients

def update(gradients, parametres, learning_rate):

    C = len(parametres) // 2

    for c in range(1, C + 1):
        parametres['W' + str(c)] = parametres['W' + str(c)] - learning_rate * gradients['dW' + str(c)]
        parametres['b' + str(c)] = parametres['b' + str(c)] - learning_rate * gradients['db' + str(c)]

    return parametres

def predict(X, parametres):
    activations = forward_propagation(X, parametres)
    C = len(parametres) // 2 
    A = activations['A' + str(C)]
    return A >= 0.5

def neural_network(X_train, y_train, X_test, y_test, hidden_layers = (32, 32, 32), learning_rate=0.1, n_iter=10000):

    np.random.seed(0)
    # Initialisation
    dimensions = list(hidden_layers)
    dimensions.insert(0, X_train.shape[0])
    dimensions.append(y_train.shape[0])
    parametres = initialisation(dimensions)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    
    # Apprentissage
    for i in tqdm(range(n_iter)):

        activations = forward_propagation(X_train, parametres)
        gradients = back_propagation(y_train, activations, parametres)
        parametres = update(gradients, parametres, learning_rate)

        if i %100 == 0:
            # Train
            C = len(parametres) // 2
            train_loss.append(log_loss(y_train, activations['A' + str(C)]))
            y_pred = predict(X_train, parametres)
            train_acc.append(accuracy_score(y_train.flatten(), y_pred.flatten()))

            # Test
            activations_test = forward_propagation(X_test, parametres)
            test_loss.append(log_loss(y_test, activations_test['A' + str(C)]))
            y_pred = predict(X_test, parametres)
            test_acc.append(accuracy_score(y_test.flatten(), y_pred.flatten()))


    plt.figure(figsize=(14, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss')
    plt.plot(test_loss, label='test loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train accuracy')
    plt.plot(test_acc, label='test accuracy')
    plt.legend()
    plt.show()

    return parametres

# dataset chien chat
X_train, y_train, X_test, y_test = load_data()

y_train = y_train.T
y_test = y_test.T

X_train = X_train.T
X_train_reshape = X_train.reshape(-1, X_train.shape[-1]) / X_train.max()

X_test = X_test.T
X_test_reshape = X_test.reshape(-1, X_test.shape[-1]) / X_test.max()

# m_train = 300
# m_test = 80
# X_test_reshape = X_test_reshape[:, :m_test]
# X_train_reshape = X_train_reshape[:, :m_train]
# y_train = y_train[:, :m_train]
# y_test = y_test[:, m_test]

parametres = neural_network(X_train_reshape, y_train, X_test_reshape, y_test)

### Generation d'un dataset de 100 lignes et 2 variables
# X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
# y = y.reshape((y.shape[0], 1))

### Generation d'un dataset en rond
# X, y = make_circles(n_samples=100, noise=0.1, factor= 0.3, random_state=0)
# X = X.T
# y = y.reshape((1,y.shape[0]))

### Testing functions
# parametres = initialisation([2, 32, 32, 1])
# activations = forward_propagation(X, parametres)
# grad = back_propagation(y, activations, parametres)
#
# for key, val in grad.items():
#     print(key, val.shape)
