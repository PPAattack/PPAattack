from tensorflow.keras.utils import to_categorical
from scipy.misc import imsave
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import time
import load_dataset


def create_model():
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape, name="conv2d_1"))
    model.add(Conv2D(64, (3, 3), activation='relu', name="conv2d_2"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax', name="dense_1"))

    model.compile(loss="categorical_crossentropy",
                  optimizer="Adadelta",
                  metrics=['accuracy'])
    model.summary()
    return model


def initial_model():
    x_train, y_train, x_test, y_test = load_dataset.get_server_dataset()
    model = create_model()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=200,
              verbose=2,
              validation_data=(x_test, y_test))
    model.save(f"./model_weights.h5")

    score = model.evaluate(x_test, y_test, verbose=2)
    print(f'Test loss:{score[0]}, Test accuracy:{score[1]}')
    print("train end\n")


def train_model(x_train, y_train, x_test, y_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    model = load_model("model_weights.h5")
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    # for j in range(10):
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=1,
              verbose=2,
              validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=2)
    print(f'Test loss:{score[0]}, Test accuracy:{score[1]}')
    print("train end\n")
    return model.get_weights()


def aggregation(weights_list):
    weights_aggregation = np.zeros(np.array(weights_list[0]).shape)
    for weights in weights_list:
        weights_aggregation = weights_aggregation + np.array(weights)
    weights_aggregation /= len(weights_list)
    weights_aggregation.tolist()
    return weights_aggregation


def train_next(weights, x_train, y_train, x_test, y_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    model = create_model()
    model.set_weights(weights)
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    # for j in range(10):
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=1,
              verbose=2,
              validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=2)
    print(f'Test loss:{score[0]}, Test accuracy:{score[1]}')
    print("train end\n")
    return model.get_weights()


def train_sgd(new_model, x, y, lr):
    old_param = new_model.get_layer("conv2d_2").get_weights()
    old_param = np.array(old_param)
    new_model.fit(x, y, epochs=1, batch_size=32, verbose=2)
    new_param = new_model.get_layer("conv2d_2").get_weights()
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


def get_input(weights):
    print("get input...")
    x_train, y_train = load_dataset.get_server_dataset_expend(200)
    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = to_categorical(y_train, 10)
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))

    Sum = []
    grad_list = []
    model = create_model()
    for j in range(20):
        sum_a = []
        grad = []
        for h in range(10):
            model.set_weights(weights)
            b = h * 1000 + 8 * j
            grad_sum_a, grad_a = train_sgd(model, x_train[b:b + 128], y_train[b:b + 128], 0.01)
            sum_a.append(grad_sum_a)
            grad.append(grad_a)
        sum_a = np.array(sum_a).reshape((10, 1))
        grad = np.array(grad)
        # print(sum_a.shape)
        Sum.append(sum_a)
        grad_list.append(grad)
    return Sum, grad_list


def get_grad(weights):
    print("get input...")
    x_train, y_train = load_dataset.get_server_dataset_expend(200)
    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = to_categorical(y_train, 10)
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))

    model = create_model()
    sum_a = []
    grad = []
    for h in range(10):
        model.set_weights(weights)
        b = h * 1000
        grad_sum_a, grad_a = train_sgd(model, x_train[b:b + 128], y_train[b:b + 128], 0.01)
        sum_a.append(grad_sum_a)
        grad.append(grad_a)
    return sum_a, grad


def judge(weights_list):
    W = [weights_list[0]]
    s0, g0 = get_grad(weights_list[0])
    s0 = np.array(s0)
    num = np.argmin(s0)
    mm = 0
    for i in range(1, len(weights_list)):
        if mm == 2:
            break
        else:
            s, ga = get_grad(weights_list[i])
            s = np.array(s)
            if num == np.argmax(s):
                W.append(weights_list[i])
                mm += 1
    if mm == 0:
        W.append(weights_list[1])
        W.append(weights_list[2])
    new_w = aggregation(W)
    return new_w


def save(path, x_attack_train, y_attack_train):
    x_attack_train = np.array(x_attack_train)
    y_attack_train = np.array(y_attack_train)
    print(x_attack_train.shape, y_attack_train.shape)
    for j, item in enumerate(x_attack_train):
        imsave(path + f"/{j}.jpg", item)


def save_data(data_path, temp_b, temp_n, n):
    x_a_n = []
    y_a_n = []
    for item_a, item_n in zip(temp_b, temp_n):
        a_temp = np.array(item_a)
        n_temp = np.array(item_n)
        temp = a_temp - n_temp
        x_a_n.append(temp)
        y_a_n.append(n)
    save_path = f"./result/test_data/" + data_path + f"/{n}"
    save(save_path, x_a_n, y_a_n)


if __name__ == '__main__':
    # initial_model()
    t0 = time.time()
    for m in range(10):
        user = load_dataset.get_client_dataset_all(200, 3000, m)
        w_list = []
        for u in user:
            w = train_model(u.x_train, u.y_train, u.x_test, u.y_test)
            w_list.append(w)
        new_weights = judge(w_list)
        g_be, g_list_be = get_input(new_weights)
        w = train_next(new_weights, user[0].x_train, user[0].y_train, user[0].x_test, user[0].y_test)
        g, g_list = get_input(w)
        path_1 = "list"
        path_2 = "num"
        save_data(path_1, g_list_be, g_list, m)
        save_data(path_2, g_be, g, m)
    t1 = time.time()
    ft = open(rf"time_test.txt", "a+")
    print(f"test:{t1 - t0}s", file=ft)
    ft.close()
