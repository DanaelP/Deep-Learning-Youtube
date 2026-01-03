import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import accuracy_score
from utilities_torch import *
from tqdm import tqdm

def initialisation(dimensions, embedding_items, device):

    parameters = {}
    g = torch.Generator().manual_seed(2147483647) # for reproducibility
    
    parameters['E'] = torch.randn((256, embedding_items), generator=g).to(device)
    C = len(dimensions)
    for c in range(1, C):
        parameters['W' + str(c)] = torch.randn((dimensions[c], dimensions[c - 1]), generator=g).to(device) * 0.01
        parameters['b' + str(c)] = torch.randn((dimensions[c], 1), generator=g).to(device) * 0.01

    return parameters

def predict(X, parameters):
    activations = forward_propagation(X, parameters)
    C = len(parameters) // 2 
    A = activations['A' + str(C)]
    return A >= 0.5

def neural_network(X_train, y_train, X_test, y_test, hidden_layers = (200,32), learning_rate=0.1, n_iter=10000, batch_size=32, embedding_items=1):

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    dimensions = list(hidden_layers)
    # dimensions.insert(0, X_train.shape[0])
    # dimensions.append(y_train.shape[0])
    dimensions.insert(0, batch_size)
    dimensions.append(batch_size)
    
    # Initialisation
    parameters = initialisation(dimensions, embedding_items, device)

    # add gradients to tensors for back_propagation
    for p in parameters:
         parameters[p].requires_grad = True

    C = len(parameters) // 2 
    train_loss = []
    # train_acc = []
    test_loss = []
    # test_acc = []

    # Trainning
    for i in tqdm(range(n_iter)):

        # Mini batch
        ix = torch.randint(0, X_train.shape[0], (batch_size,), device=device)

        X_train_long = X_train.long()
        y_train_long = y_train.squeeze().long()
        X_test_long = X_test.long()
        y_test_long = y_test.squeeze().long()

        # print(y_train_long.shape)

        emb = parameters['E'][X_train_long[ix]]
        emb = emb.view(-1, 4096*embedding_items)

        # if i >= 10000:
        #     learning_rate=0.01

        # init activation
        activations = {'A0' : emb}
        logits = {}
        # forward pass
        for c in range(1, C + 1):
            # print('parameters: ', parameters['W' + str(c)].shape)
            # print('activations: ', activations['A' + str(c-1)].shape)
            logits['Z' + str(c)] = parameters['W' + str(c)] @ activations['A' + str(c - 1)] + parameters['b' + str(c)]
            activations['A' + str(c)] = torch.sigmoid(logits['Z' + str(c)])
        loss = F.cross_entropy(logits['Z' + str(C)], y_train_long[ix])

        # if i %10 == 0:
        train_loss.append(loss.log10().item())

            # emb_test = parameters['E'][X_test_long[ixt]]
            # emb_test = emb_test.view(-1, 4096*1)
            # activations_test = {'A0' : emb_test}
            # for c in range(1, C + 1):
            #     activations_test['A' + str(c)] = torch.sigmoid(parameters['W' + str(c)] @ activations_test['A' + str(c - 1)] + parameters['b' + str(c)])
            # test_lossi = F.cross_entropy(logits['Z' + str(C)], y_test_long[ixt])
            # test_loss.append(test_lossi.log10().item())



        # backward pass
        for p in parameters:
            parameters[p].grad = None
        loss.backward()

        for c in range(1, C + 1):
            parameters['W' + str(c)].data += - learning_rate * parameters['W' + str(c)].grad
            parameters['b' + str(c)].data += - learning_rate * parameters['b' + str(c)].grad


        # break

    print(loss.log10().item())
    plt.plot(train_loss, label='train loss')
    # plt.plot(test_loss, label='test loss')
    plt.legend()
    plt.savefig('graph.png')
    
    ixt = torch.randint(0, X_test.shape[0], (batch_size,), device=device)
    emb_test = parameters['E'][X_test_long[ixt]]
    emb_test = emb_test.view(-1, 4096*embedding_items)
    activations_test = {'A0' : emb_test}
    for c in range(1, C + 1):
        activations_test['A' + str(c)] = torch.sigmoid(parameters['W' + str(c)] @ activations_test['A' + str(c - 1)] + parameters['b' + str(c)])
    test_lossi = F.cross_entropy(logits['Z' + str(C)], y_test_long[ixt])
    # test_loss.append(test_lossi.log10().item())
    print(test_lossi.log10().item())

    #
    #     # forward propagation
    #     logits = 
    #     activations = forward_propagation(X_train, parameters)
    #     gradients = back_propagation(y_train, activations, parameters)
    #     parameters = update(gradients, parameters, learning_rate)
    #
    #     if i %100 == 0:
    #         # train_loss.append(log_loss(y_train, activations['A' + str(C)]))
    #         train_loss.append(F.cross_entropy(activations['A' + str(C)], y_train))
    #         y_pred = predict(X_train, parameters)
    #         train_acc.append(accuracy_score(y_train.flatten(), y_pred.flatten()))
    #
    #         # Test
    #         activations_test = forward_propagation(X_test, parameters)
    #         test_loss.append(log_loss(y_test, activations_test['A' + str(C)]))
    #         y_pred = predict(X_test, parameters)
    #         test_acc.append(accuracy_score(y_test.flatten(), y_pred.flatten()))
    #
    #
    # plt.figure(figsize=(14, 4))
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(train_loss, label='train loss')
    # plt.plot(test_loss, label='test loss')
    # plt.legend()
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(train_acc, label='train accuracy')
    # plt.plot(test_acc, label='test accuracy')
    # plt.legend()
    # plt.show()
    #
    # return parameters

# def initialisation(dimensions):
#
#     parameters = {}
#     C = len(dimensions)
#
#     for c in range(1, C):
#         parameters['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
#         parameters['b' + str(c)] = np.random.randn(dimensions[c], 1)
#
#     return parameters

# def forward_propagation(X, parameters):
#
#     activations = {'A0' : X}
#     C = len(parameters) // 2
#
#     for c in range(1, C + 1):
#         Z = parameters['W' + str(c)].dot(activations['A' + str(c - 1)]) + parameters['b' + str(c)]
#         activations['A' + str(c)] = 1 / (1 + np.exp(-Z))
#
#     return activations

# def log_loss(y, A):
#     epsilon = 1e-15
#     return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))

# def back_propagation(y, activations, parameters):
#
#     m = y.shape[1]
#     C = len(parameters) // 2
#
#     dZ = activations['A' + str(C)] - y
#     gradients = {}
#
#     for c in reversed(range(1, C + 1)):
#         gradients['dW' + str(c)] = 1 / m * np.dot(dZ, activations['A' + str(c - 1)].T)
#         gradients['db' + str(c)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
#         if c > 1:
#             dZ = np.dot(parameters['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])
#
#     return gradients

# def update(gradients, parameters, learning_rate):
#
#     C = len(parameters) // 2
#
#     for c in range(1, C + 1):
#         parameters['W' + str(c)] = parameters['W' + str(c)] - learning_rate * gradients['dW' + str(c)]
#         parameters['b' + str(c)] = parameters['b' + str(c)] - learning_rate * gradients['db' + str(c)]
#
#     return parameters

# def predict(X, parameters):
#     activations = forward_propagation(X, parameters)
#     C = len(parameters) // 2 
#     A = activations['A' + str(C)]
#     return A >= 0.5

# def neural_network(X_train, y_train, X_test, y_test, hidden_layers = (32, 32, 32), learning_rate=0.1, n_iter=10000):
#
#     np.random.seed(0)
#     # Initialisation
#     dimensions = list(hidden_layers)
#     dimensions.insert(0, X_train.shape[0])
#     dimensions.append(y_train.shape[0])
#     parameters = initialisation(dimensions)
#
#     train_loss = []
#     train_acc = []
#     test_loss = []
#     test_acc = []
#
#     # Apprentissage
#     for i in tqdm(range(n_iter)):
#
#         activations = forward_propagation(X_train, parameters)
#         gradients = back_propagation(y_train, activations, parameters)
#         parameters = update(gradients, parameters, learning_rate)
#
#         if i %100 == 0:
#             # Train
#             C = len(parameters) // 2
#             train_loss.append(log_loss(y_train, activations['A' + str(C)]))
#             y_pred = predict(X_train, parameters)
#             train_acc.append(accuracy_score(y_train.flatten(), y_pred.flatten()))
#
#             # Test
#             activations_test = forward_propagation(X_test, parameters)
#             test_loss.append(log_loss(y_test, activations_test['A' + str(C)]))
#             y_pred = predict(X_test, parameters)
#             test_acc.append(accuracy_score(y_test.flatten(), y_pred.flatten()))
#
#
#     plt.figure(figsize=(14, 4))
#
#     plt.subplot(1, 2, 1)
#     plt.plot(train_loss, label='train loss')
#     plt.plot(test_loss, label='test loss')
#     plt.legend()
#
#     plt.subplot(1, 2, 2)
#     plt.plot(train_acc, label='train accuracy')
#     plt.plot(test_acc, label='test accuracy')
#     plt.legend()
#     plt.show()
#
#     return parameters

# dataset chien chat
X_train, y_train, X_test, y_test = load_data()

y_train = y_train
y_test = y_test

X_train_reshape = X_train.view(-1, 4096).float()
X_test_reshape = X_test.view(-1, 4096).float()


# m_train = 300
# m_test = 80
# X_test_reshape = X_test_reshape[:, :m_test]
# X_train_reshape = X_train_reshape[:, :m_train]
# y_train = y_train[:, :m_train]
# y_test = y_test[:, m_test]

parameters = neural_network(X_train_reshape, y_train, X_test_reshape, y_test)

### Generation d'un dataset de 100 lignes et 2 variables
# X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
# y = y.reshape((y.shape[0], 1))

### Generation d'un dataset en rond
# X, y = make_circles(n_samples=100, noise=0.1, factor= 0.3, random_state=0)
# X = X.T
# y = y.reshape((1,y.shape[0]))

### Testing functions
# parameters = initialisation([2, 32, 32, 1])
# activations = forward_propagation(X, parameters)
# grad = back_propagation(y, activations, parameters)
#
# for key, val in grad.items():
#     print(key, val.shape)
