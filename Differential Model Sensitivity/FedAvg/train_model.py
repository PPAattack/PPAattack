import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils


def create_model():
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape, name="conv2d_1"))
    model.add(Conv2D(64, (3, 3), activation='relu', name="conv2d_11"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', name="dense_11"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax', name="dense_1"))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.summary()
    return model


def split_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_data = []
    y_data = []
    num = 0
    for i in range(10):
        for j in range(60000):
            if y_train[j] == i:
                x_data.append(x_train[j])
                y_data.append(y_train[j])
                num += 1
        # print(num)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    print(x_data.shape)
    print(y_data.shape)
    return x_data, y_data, x_test, y_test


def load_dataset(target, target_num, flag):
    x, y, x_, y_ = split_dataset()
    x_data = []
    y_data = []
    for i in range(10):
        if i != target:
            for j in range((T - target_num) // 9):
                x_data.append(x[N[i] + j + flag])
                y_data.append(y[N[i] + j + flag])
        else:
            for j in range(target_num):
                x_data.append(x[N[i] + j + flag])
                y_data.append(y[N[i] + j + flag])
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    print(x_data.shape)
    print(y_data.shape)
    return x_data, y_data, x_, y_


def initial_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[:6000]
    y_train = y_train[:6000]
    model = create_model()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=15,
              verbose=2,
              validation_data=(x_test, y_test))
    model.save(f"./model_weights.h5")

    score = model.evaluate(x_test, y_test, verbose=2)
    print(f'Test loss:{score[0]}, Test accuracy:{score[1]}')
    print("train end\n")


def train_model(x_train, y_train, x_test, y_test, num):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    model = load_model("model_weights.h5")
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=1,
              verbose=2,
              validation_data=(x_test, y_test))
    model.save(f"./model/{num}_weights.h5")

    score = model.evaluate(x_test, y_test, verbose=2)
    print(f'Test loss:{score[0]}, Test accuracy:{score[1]}')
    print("train end\n")


def load():
    model_list = []
    for i in range(3):
        m = load_model(f"./model/{i}_weights.h5")
        model_list.append(m)
    return model_list


def train_next(x_train, y_train, x_test, y_test, model):
    model_n = create_model()
    w = model.get_weights()
    model_n.set_weights(w)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    model_n.fit(x_train, y_train,
                batch_size=128,
                epochs=1,
                verbose=2,
                validation_data=(x_test, y_test))
    # model.save(f"./model_train/{target}/next_weights.h5")

    score = model_n.evaluate(x_test, y_test, verbose=2)
    print(f'Test loss:{score[0]}, Test accuracy:{score[1]}')
    print("train end\n")
    return model_n


def aggregation(participant_model):
    model_average = create_model()
    weights = np.zeros(np.array(participant_model[0].get_weights()).shape)
    for m in participant_model:
        w = m.get_weights()
        weights = weights + np.array(w)
    weights = weights / len(participant_model)
    weights.tolist()
    model_average.set_weights(weights)
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_test = x_test[:5000]
    # y_test = y_test[:5000]
    # x_test = x_test.astype('float32')
    # x_test /= 255
    # y_test = np_utils.to_categorical(y_test, 10)
    # x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    # score = model_average.evaluate(x_test, y_test, verbose=2)
    # print(f'Test loss:{score[0]}, Test accuracy:{score[1]}', file=f)
    return model_average


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


def load_dataset_2(n, x_, y_, num):
    x_train = []
    y_train = []
    for i in range(10):
        t = 0
        for j in range(num):
            if y_[j] == i:
                x_train.append(x_[j])
                y_train.append(y_[j])
                t += 1
                if t == n:
                    break
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    print(x_train.shape)
    print(y_train.shape)
    return x_train, y_train


def get_input(model):
    print("get input...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = load_dataset_2(1000, x_train[20000:40000], y_train[20000:40000], 20000)
    # x_test, y_test = load_dataset_2(100, x_test, y_test, 10000)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = np_utils.to_categorical(y_train, 10)
    # y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    # x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    Sum = []
    grad_list = []
    a_model = create_model()
    w = model.get_weights()
    for j in range(5):
        sum_a = []
        grad = []
        for h in range(10):
            a_model.set_weights(w)
            # b = random.randint(h * 5000, (h + 1) * 5000 - 129)
            b = h * 1000 + 80 * j
            grad_sum_a, grad_a = train_sgd(a_model, x_train[b:b + 128], y_train[b:b + 128], 0.01)
            sum_a.append(grad_sum_a)
            grad.append(grad_a)
        sum_a = np.array(sum_a).reshape((10, 1))
        grad = np.array(grad)
        # print(sum_a.shape)
        Sum.append(sum_a)
        grad_list.append(grad)
    return Sum, grad_list


if __name__ == '__main__':
    initial_model()
    epoch = 5
    T = 6000
    N = [0, 5923, 12665, 18623, 24754, 30596, 36017, 41935, 48200, 54051]
    x_1, y_1, x_1_t, y_1_t = load_dataset(1, 3000, 0)
    x_2, y_2, x_2_t, y_2_t = load_dataset(1, 600, 1000)
    x_3, y_3, x_3_t, y_3_t = load_dataset(1, 600, 2000)
    train_model(x_1, y_1, x_1_t, y_1_t, 0)
    train_model(x_2, y_2, x_2_t, y_2_t, 1)
    train_model(x_3, y_3, x_3_t, y_3_t, 2)
    model_l = load()
    for e in range(epoch):
        f = open(f"result.txt", "a+")
        # print(f"epoch:{e}", file=f)
        model_a = aggregation(model_l)
        g_a, g_a_list = get_input(model_a)
        print(f"g_a:{g_a}", file=f)
        model_next = train_next(x_1, y_1, x_1_t, y_1_t, model_a)
        model_next_1 = train_next(x_2, y_2, x_2_t, y_2_t, model_a)
        model_next_2 = train_next(x_3, y_3, x_3_t, y_3_t, model_a)
        model_l = [model_next, model_next_1, model_next_2]
        g_a_n, g_a_list_n = get_input(model_next)
        print(f"g_a_n:{g_a_n}", file=f)
        f.close()
