# %%
##import libraries
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random

data = sio.loadmat("/Users/tab/Project/AI/machine_learning_alg/mnist_01.mat")

## list all keys
keys = data.keys() # 'X_test', 'X_train', 'label_test', 'label_train'

X_train = data['X_train'] # shape (10000, 784)
X_test = data['X_test']
label_train = data['label_train'] # shape (10000, 1)
label_test = data['label_test']
cnt_train = len(label_train)
cnt_test = len(label_test)
cnt_dim = len(X_train[0])
fig = plt.figure(figsize=(28,28))
row = 2
col = 5
print(X_train[0].shape)
for i in range(10):
    rand = random.randint(1, cnt_train)
    print(label_train[rand][0])
    #reshape to 28*28 and normal rgb from 0~255 to 0~1
    sub = fig.add_subplot(row, col, i+1)
    sub.title.set_text(label_train[rand][0])
    plt.imshow(X_train[rand].reshape((28, 28)) / 255.0)
# plt.show()
y_train = np.zeros(cnt_train)
y_test = np.zeros(cnt_test)

for i in range(cnt_train):
    y_train[i] = 1 if label_train[i][0] == 1 else -1
for i in range(cnt_test):
    y_test[i] = 1 if label_test[i][0] == 1 else -1


# %%
def accuracy(w, X, y):
    cnt_correct = 0
    fig = plt.figure(figsize=(28,28))
    for i in range(len(X)):
        # t = i
        # i = random.randint(1, len(X))
        test = y[i]*X[i].dot(w)
        if y[i]*X[i].dot(w) > 0:
            cnt_correct = cnt_correct+1
            # sub = fig.add_subplot(5, 5, t+1)
            # sub.title.set_text(y[i])
            # plt.imshow(X[i].reshape((28, 28)))
    # plt.show()
    print("accuracy is %f", cnt_correct/len(X))
    

# %%
# SGD
eta = 0.01
hyper_parameter = 1
X_train_float = X_train/255.0
X_test_float = X_test/255.0
dw1 = - X_train_float.transpose().dot(y_train) / cnt_train
w = np.zeros(cnt_dim)
iter_max = 6
range_iter = range(iter_max)
fws = np.zeros(iter_max)
for t in range_iter:
    rand = random.randint(1, 2)
    if rand == 1:
        for i in range(cnt_train):
            if (y_train[i]*X_train_float[i].dot(w)<1):
                w = w - eta * dw1
    else:
        w = w - eta * hyper_parameter * w
    # fws[t] = y_train.transpose().dot(X_test_float.dot(w))
    # plt.plot(range_iter)
# %%
# accuracy(w, X_test_float, y_test)
t = (X_train_float.dot(w))
print(y_train.dot(t))