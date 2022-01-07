import math
import random
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from scipy.misc import imsave


def create_model():
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape, name="conv2d_1"))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax', name="dense_1"))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.summary()
    return model


def initial_model(train_x, train_y, test_x, test_y):
    model = create_model()
    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    train_x /= 255
    test_x /= 255
    train_y = np_utils.to_categorical(train_y, 10)
    test_y = np_utils.to_categorical(test_y, 10)
    train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
    test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))
    model.fit(train_x, train_y,
              batch_size=128,
              epochs=15,
              verbose=2,
              validation_data=(test_x, test_y))
    model.save(f"./model_weights_aggregation.h5")
    score = model.evaluate(test_x, test_y, verbose=2)
    print(f'Test loss:{score[0]}, Test accuracy:{score[1]}')
    print("train end\n")


def train(train_x, train_y, test_x, test_y):
    model = create_model()
    m = load_model(f"model_weights_aggregation.h5")
    w = m.get_weights()
    model.set_weights(w)
    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    train_x /= 255
    test_x /= 255
    train_y = np_utils.to_categorical(train_y, 10)
    test_y = np_utils.to_categorical(test_y, 10)
    train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
    test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))
    model.fit(train_x, train_y,
              batch_size=128,
              epochs=1,
              verbose=2,
              validation_data=(test_x, test_y))
    score = model.evaluate(test_x, test_y, verbose=2)
    print(f'Test loss:{score[0]}, Test accuracy:{score[1]}')
    print("train end\n")
    # weights = add_noise(model.get_weights())
    weights = model.get_weights()
    return weights


def train_next(train_x, train_y, test_x, test_y, m):
    model = create_model()
    w = m.get_weights()
    model.set_weights(w)
    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    train_x /= 255
    test_x /= 255
    train_y = np_utils.to_categorical(train_y, 10)
    test_y = np_utils.to_categorical(test_y, 10)
    train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
    test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))
    model.fit(train_x, train_y,
              batch_size=128,
              epochs=1,
              verbose=2,
              validation_data=(test_x, test_y))
    score = model.evaluate(test_x, test_y, verbose=2)
    print(f'Test loss:{score[0]}, Test accuracy:{score[1]}')
    print("train end\n")
    # weights = add_noise(model.get_weights())
    weights = model.get_weights()
    return weights


def distribute_dataset_ini(train_x, train_y, count):
    num_list = [0, 5923, 12665, 18623, 24754, 30596, 36017, 41935, 48200, 54051, 60000]
    X = []
    Y = []
    for i in range(10):
        for j in range(count):
            X.append(train_x[num_list[i] + j])
            Y.append(train_y[num_list[i] + j])
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape)
    print(Y.shape)
    return X, Y


def distribute_dataset_get(train_x, train_y, count):
    num_list = [0, 5923, 12665, 18623, 24754, 30596, 36017, 41935, 48200, 54051, 60000]
    X = []
    Y = []
    for i in range(10):
        for j in range(count):
            X.append(train_x[num_list[i] + j % 400])
            Y.append(train_y[num_list[i] + j % 400])
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape)
    print(Y.shape)
    return X, Y


def distribute_dataset_a(train_x, train_y, num, count):
    num_list = [0, 5923, 12665, 18623, 24754, 30596, 36017, 41935, 48200, 54051, 60000]
    X = []
    Y = []
    for i in range(10):
        if i == num:
            for j in range(count):
                X.append(train_x[num_list[i] + j % 400])
                Y.append(train_y[num_list[i] + j % 400])
        else:
            min_num = 300 + random.randint(-50, 50)
            # print(f"{i}_min:{min_num}", file=f)
            for j in range(min_num):
                X.append(train_x[num_list[i] + j % 400])
                Y.append(train_y[num_list[i] + j % 400])
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape)
    print(Y.shape)
    return X, Y


def distribute_dataset_1(train_x, train_y, num):
    num_list = [0, 5923, 12665, 18623, 24754, 30596, 36017, 41935, 48200, 54051, 60000]
    X = []
    Y = []
    for i in range(10):
        if i != num:
            min_num = 800 + random.randint(-50, 50)
            # print(f"{i}_min:{min_num}", file=f)
            for j in range(min_num):
                X.append(train_x[num_list[i] + j % 400])
                Y.append(train_y[num_list[i] + j % 400])
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape)
    print(Y.shape)
    return X, Y


def distribute_dataset_2(train_x, train_y, num):
    num_list = [0, 5923, 12665, 18623, 24754, 30596, 36017, 41935, 48200, 54051, 60000]
    X = []
    Y = []
    for i in range(10):
        if i != num:
            min_num = 800 + random.randint(-50, 50)
            # print(f"{i}_min:{min_num}", file=f)
            for j in range(min_num):
                X.append(train_x[num_list[i] + j % 400])
                Y.append(train_y[num_list[i] + j % 400])
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape)
    print(Y.shape)
    return X, Y


def load_dataset():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    X = []
    Y = []
    t = 0
    for i in range(10):
        for j in range(60000):
            if train_y[j] == i:
                X.append(train_x[j])
                Y.append(train_y[j])
                t += 1
        print(t)
    return X, Y, test_x, test_y


def aggregation(participant_model):
    model_average = create_model()
    weights = np.zeros(np.array(participant_model[0]).shape)
    for w in participant_model:
        weights = weights + np.array(w)
    weights = weights / len(participant_model)
    weights.tolist()
    model_average.set_weights(weights)
    return model_average


def test(train_x, train_y, test_x, test_y, weights):
    model = create_model()
    model.set_weights(weights)
    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    train_x /= 255
    test_x /= 255
    train_y = np_utils.to_categorical(train_y, 10)
    test_y = np_utils.to_categorical(test_y, 10)
    train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
    test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))
    model.fit(train_x, train_y,
              batch_size=128,
              epochs=1,
              verbose=2,
              validation_data=(test_x, test_y))
    score = model.evaluate(test_x, test_y, verbose=2)
    f = open("acc.txt", "a+")
    print(f'Test loss:{score[0]}, Test accuracy:{score[1]}', file=f)
    f.close()


def train_sgd(new_model, x, y, lr):
    old_param = new_model.get_layer("conv2d_1").get_weights()
    old_param = np.array(old_param)
    new_model.fit(x, y, epochs=10, batch_size=32, verbose=2)
    new_param = new_model.get_layer("conv2d_1").get_weights()
    new_param = np.array(new_param)
    grad = 0
    grad_list = []
    for (item_old, item_new) in zip(old_param, new_param):
        for i in range(len(item_old)):
            old_param_line = item_old[i].flatten()
            new_param_line = item_new[i].flatten()
            tep = (old_param_line - new_param_line) / lr
            for j in tep:
                grad += abs(j)
                grad_list.append(j)
    # grad_list = np.array(grad_list)
    # print(grad_list.shape)
    # print(grad)
    return grad, grad_list


def get_flat(weight):
    weight_list = []
    for item in weight:
        for i in range(len(item)):
            for j in item[i].flatten():
                weight_list.append(j)
    return weight_list


def get_input(train_x, train_y, w):
    print("get input...")
    train_x = train_x.astype('float32')
    train_x /= 255
    train_y = np_utils.to_categorical(train_y, 10)
    train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))

    Sum = []
    grad_list = []
    a_model = create_model()
    for j in range(100):
        sum_a = []
        grad = []
        for h in range(10):
            a_model.set_weights(w)
            # b = random.randint(h * 5000, (h + 1) * 5000 - 129)
            b = h * 1000 + 8 * j
            grad_sum_a, grad_a = train_sgd(a_model, train_x[b:b + 128], train_y[b:b + 128], 0.01)
            sum_a.append(grad_sum_a)
            grad.append(grad_a)
        sum_a = np.array(sum_a).reshape((10, 1))
        grad = np.array(grad)
        # print(sum_a.shape)
        Sum.append(sum_a)
        grad_list.append(grad)
    return Sum, grad_list


def save(path, x_attack_train, y_attack_train):
    x_attack_train = np.array(x_attack_train)
    y_attack_train = np.array(y_attack_train)
    print(x_attack_train.shape, y_attack_train.shape)
    for j, item in enumerate(x_attack_train):
        imsave(path + f"/{j}.jpg", item)


def save_data(data_path, temp_b, temp_n):
    x_a_n = []
    y_a_n = []
    for item_a, item_n in zip(temp_b, temp_n):
        a_temp = np.array(item_a)
        n_temp = np.array(item_n)
        temp = a_temp - n_temp
        x_a_n.append(temp)
        y_a_n.append(k)
    save_path = f"./result/data_train/" + data_path + f"/{k}"
    save(save_path, x_a_n, y_a_n)


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_dataset()
    x_train_ini, y_train_ini = distribute_dataset_ini(x_train, y_train, 400)
    initial_model(x_train_ini, y_train_ini, x_test, y_test)
    f = open("num_train.txt", "a+")
    for k in range(10):
        n_max = 4000 + random.randint(-50, 50)
        print(f"{k}_max:{n_max}", file=f)
        x_train_a, y_train_a = distribute_dataset_a(x_train, y_train, k, n_max)
        x_train_1, y_train_1 = distribute_dataset_1(x_train, y_train, k)
        x_train_2, y_train_2 = distribute_dataset_2(x_train, y_train, k)
        w_a = train(x_train_a, y_train_a, x_test, y_test)
        w_1 = train(x_train_1, y_train_1, x_test, y_test)
        w_2 = train(x_train_2, y_train_2, x_test, y_test)
        w_list = [w_a, w_1, w_2]
        model_a = aggregation(w_list)
        model_a.save(f"./result/model_train/{k}/model_weights_aggregation.h5")
        x_train_get, y_train_get = distribute_dataset_get(x_train, y_train, 1000)
        g_a, g_a_list = get_input(x_train_get, y_train_get, model_a.get_weights())
        model_next = train_next(x_train_a, y_train_a, x_test, y_test, model_a)
        g_n, g_n_list = get_input(x_train_get, y_train_get, model_next)
        path_1 = "list"
        path_2 = "num"
        save_data(path_1, g_a_list, g_n_list)
        save_data(path_2, g_a, g_n)
    f.close()

