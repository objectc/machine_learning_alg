# %%
##import libraries
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random

# read input data
data = sio.loadmat("mnist_01.mat")

## list all keys
keys = data.keys() # 'X_test', 'X_train', 'label_test', 'label_train'

X_train = data['X_train'] # shape (10000, 784)
X_test = data['X_test']
label_train = data['label_train'] # shape (10000, 1)
label_test = data['label_test']
cnt_train = len(label_train)
cnt_test = len(label_test)
cnt_dim = len(X_train[0])

row = 2
col = 5
print(X_train[0].shape)
# data preparation
for i in range(10):
    rand = random.randint(1, cnt_train)
    print(label_train[rand][0])
#     reshape to 28*28 and normal rgb from 0~255 to 0~1
#     sub = fig.add_subplot(row, col, i+1)
#     sub.title.set_text(label_train[rand][0])
#     plt.imshow(X_train[rand].reshape((28, 28)) / 255.0)
# plt.show()
y_train = np.zeros(cnt_train)
y_test = np.zeros(cnt_test)
# "0" to "-1"
for i in range(cnt_train):
    y_train[i] = 1 if label_train[i][0] == 1 else -1
for i in range(cnt_test):
    y_test[i] = 1 if label_test[i][0] == 1 else -1


# %%
# compute accuracy function
def accuracy(w, X, y):
    cnt_correct = 0
    # fig = plt.figure(figsize=(28,28))
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
    return cnt_correct/len(X)

# visualize pair element 0 and 1
def visualize(w, X, y):
    pos = np.zeros(cnt_dim)
    neg = np.zeros(cnt_dim)
    cnt_pos = 0
    cnt_neg = 0
# def accuracy(w, X, y):
    for i in range(len(X)):
        test = y[i]*X[i].dot(w)
        if y[i]*X[i].dot(w) > 0:
            if y[i]>0:
                pos = pos + X[i]
                cnt_pos = cnt_pos+1
            else:
                neg = neg + X[i]
                cnt_neg = cnt_neg+1
    # remember to mask 0 as transparent
    neg = np.ma.masked_where(neg < 1, neg)
    pos = np.ma.masked_where(pos < 1, pos)
    plt.imshow(neg.reshape(28, 28).transpose()/cnt_neg, cmap='Blues_r')
    plt.imshow(pos.reshape(28, 28).transpose()/cnt_pos, cmap='Reds_r')

# %%
# SGD
eta = 0.01
hyper_parameter = [10**(-6) , 10**(-3) , 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500, 1000, 10000]
# hyper_parameter = [10**(-6) , 10**(-3) , 0.1, 0.5, 1, 2, 5, 10, 20, 50]
# hyper_parameter = [1]
X_train_float = X_train/255.0
X_test_float = X_test/255.0
dw1 = -(X_train_float.transpose()*y_train).transpose()
iter_max = 10000
cnt_sample = 100
interval = int(iter_max/cnt_sample)
fws = np.zeros(100)
t_list = np.zeros(100)
frac_t_list = np.zeros(100)
accuracy_list_test = []
accuracy_list_train = []
w_rho_list = []

# rho = [0, 0.01, 0.1, 0.2, 0.3, 0.5, 0.7]
# rho = [0.01, 0.5, 0.7]
rho = [0]
# y_train_noise = np.zeros(cnt_train)
for noise_rate in rho:
    y_train_noise = y_train
    accuracy_test = []
    accuracy_train = []
    w_rho = []
    noise_int = noise_rate * 100
    for noise_iter in range(cnt_train):
        rand = random.randint(0, noise_int)
        if rand < noise_rate*noise_int:
            y_train_noise[noise_iter] = -y_train_noise[noise_iter]

    for lambda_iter in hyper_parameter:
        fws = np.zeros(100)
        dw1 = -(X_train_float.transpose()*y_train_noise).transpose()
        print("lambda: ",lambda_iter)
        w = np.zeros(cnt_dim)
        for t in range(iter_max):
            # eta_t = 1/((t+1)**2)
            # eta_t = eta
            eta_t = 1/(t+1)
            rand = random.randint(0, cnt_train-1)
            # if rand == 1:
                # for i in range(cnt_train):
            if (y_train_noise[rand]*X_train_float[rand].dot(w)<1):
                w = w - eta_t * (dw1[rand]+lambda_iter * w)
            
            # if t % interval == 0:
            #     k = int(t/interval)
            #     for i in range(cnt_train):
            #         fws[k] = fws[k] + max(1-y_train_noise[i]*X_train_float[i].dot(w),0)
            #     fws[k] = fws[k]/cnt_train + lambda_iter/2*w.transpose().dot(w)
            #     print(fws[k])
            #     t_list[k] = t
            #     frac_t_list[k] = float(1/(t+1))
        accuracy_train.append(accuracy(w, X_train_float, y_train))
        accuracy_test.append(accuracy(w, X_test_float, y_test))
        w_rho.append(w.transpose().dot(w)/cnt_dim)
    x = range(0,len(hyper_parameter))
    plt.subplot(2, 1, 1)
    plt.xticks(x, hyper_parameter)
    plt.plot(x, accuracy_train, 'r-')
    plt.plot(x, accuracy_test, 'g-')
    plt.subplot(2, 1, 2)
    plt.xticks(x, hyper_parameter)
    plt.plot(x, w_rho, 'b-')
    # plt.plot(range(hyper_parameter), accuracy_train, 'r-')
    # plt.plot(range(hyper_parameter), accuracy_test, 'g-')
    # plt.plot(range(hyper_parameter), w_rho, 'b-')
    plt.show()
        # plt.plot(range(0, iter_max, interval), fws, 'r-')
        # plt.plot(range(0, iter_max, interval), t_list, 'g-')
        # plt.plot(range(0, iter_max, interval), frac_t_list, 'b-')
        # plt.legend(['F(w)', '1/t'], loc='upper right')
        # plt.show()
    # hs_params = [10, 20, 50, 100, 200, 400]

    # fig = plt.figure(figsize=(28,28))
    # for index, top in enumerate(hs_params):
    #     threshold = sorted(w, key=lambda value: np.abs(value), reverse=True)[top]
    #     hs_w = np.array(w, copy=True)
    #     hs_w[np.abs(hs_w)<threshold] = 0

    #     sub = fig.add_subplot(2, 3, index+1)
    #     sub.title.set_text(top)

    #     visualize(hs_w, X_test_float, y_test)
    # plt.show()
            
    accuracy_list_test.append(accuracy_test)
    accuracy_list_train.append(accuracy_train)
    w_rho_list.append(w_rho)

# plt.plot(hyper_parameter, accuracy_list, 'g^')
# plt.show()
print("accuracy_list_train", accuracy_list_train)
print("accuracy_list_test", accuracy_list_test)
print("w_rho", w_rho_list)
 

