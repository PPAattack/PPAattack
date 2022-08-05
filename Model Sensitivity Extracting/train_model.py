import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from scipy.misc import imsave


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


def load_dataset(main, main_num, vice, vice_num, other):
    class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_data = []
    y_data = []
    num_main, num_vice = 0, 0
    for i in range(60000):
        if y_train[i] == main:
            x_data.append(x_train[i])
            y_data.append(y_train[i])
            num_main += 1
        if num_main == main_num:
            break
    for i in range(60000):
        if y_train[i] == vice:
            x_data.append(x_train[i])
            y_data.append(y_train[i])
            num_vice += 1
        if num_vice == vice_num:
            break
    for c in class_list:
        num_other = 0
        if c != main and c != vice:
            for i in range(60000):
                if y_train[i] == c:
                    x_data.append(x_train[i])
                    y_data.append(y_train[i])
                    num_other += 1
                if num_other == other:
                    break
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    return x_data, y_data


def initial_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[:2000]
    y_train = y_train[:2000]
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
              epochs=50,
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
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    model = load_model("model_weights.h5")
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=5,
              verbose=2,
              validation_data=(x_test, y_test))
    model.save(f"./model/{k}/{num}_weights.h5")

    score = model.evaluate(x_test, y_test, verbose=2)
    print(f'Test loss:{score[0]}, Test accuracy:{score[1]}')
    print("train end\n")
    return model


def load():
    model_list = []
    for i in range(3):
        m = load_model(f"./model/{k}/{i}_weights.h5")
        model_list.append(m)
    return model_list


def aggregation(participant_model):
    model_average = create_model()
    weights = np.zeros(np.array(participant_model[0].get_weights()).shape)
    for m in participant_model:
        w = m.get_weights()
        weights = weights + np.array(w)
    weights = weights / len(participant_model)
    weights.tolist()
    model_average.set_weights(weights)
    return model_average


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


def train_sgd(new_model, x, y, lr):
    old_param = new_model.get_layer("conv2d_2").get_weights()
    old_param = np.array(old_param)
    new_model.fit(x, y, epochs=10, batch_size=32, verbose=2)
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


def get_flat(weight):
    weight_list = []
    for item in weight:
        for i in range(len(item)):
            for j in item[i].flatten():
                weight_list.append(j)
    return weight_list


def get_input(model):
    print("get input...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = load_dataset_2(1000, x_train[20000:40000], y_train[20000:40000], 20000)
    # x_test, y_test = load_dataset_2(100, x_test, y_test, 10000)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, 10)
    # y_test = to_categorical(y_test, 10)
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


def train_next(x_train, y_train, x_test, y_test, model, num):
    model_n = create_model()
    w = model.get_weights()
    model_n.set_weights(w)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    model_n.fit(x_train, y_train,
                batch_size=128,
                epochs=1,
                verbose=2,
                validation_data=(x_test, y_test))
    model.save(f"./model/{k}/{num}_next_weights.h5")

    score = model_n.evaluate(x_test, y_test, verbose=2)
    print(f'Test loss:{score[0]}, Test accuracy:{score[1]}')
    print("train end\n")
    return model_n


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
    save_path = f"./result/" + data_path + f"/{k}"
    save(save_path, x_a_n, y_a_n)


if __name__ == '__main__':
    initial_model()
    for k in range(10):
        if k == 0:
            k_v = 9
        else:
            k_v = k - 1
        user0_x, user0_y = load_dataset(k, 2000, k_v, 440, 320)
        # user1_x, user1_y = load_dataset(k, 100, k_v, 1100, 1100)
        # user2_x, user2_y = load_dataset(k, 100, k_v, 1100, 1100)
        m = train_model(user0_x[:5000], user0_y[:5000], user0_x[4000:], user0_y[4000:], 0)
        # train_model(user1_x[:8000], user1_y[:8000], user1_x[8000:], user1_y[8000:], 1)
        # train_model(user2_x[:8000], user2_y[:8000], user2_x[8000:], user2_y[8000:], 2)
        # model_l = load()
        # model_a = aggregation(model_l)
        # g_a, g_a_list = get_input(model_a)
        # model_next = train_next(user0_x[:8000], user0_y[:8000], user0_x[8000:], user0_y[8000:], model_a, 0)
        # g_n, g_n_list = get_input(model_next)
        # path_1 = "list"
        # path_2 = "num"
        # save_data(path_1, g_a_list, g_n_list)
        # save_data(path_2, g_a, g_n)
        g_a, g_a_list = get_input(m)
        f = open(fr"./result/{k}/result.txt", "w")
        for item in g_a:
            print(item, file=f)
        f.close()

