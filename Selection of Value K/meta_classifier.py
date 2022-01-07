from scipy.misc import imsave
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.image as img
from keras.utils import np_utils


def create_attack_model_2():
    input_shape = (10, 1, 1)
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    # model.add(Dense(10, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.summary()
    return model


def create_attack_model():
    input_shape = (10, 320, 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.summary()
    return model


def load_train_data(data_path):
    print("load data...")
    x_train_attack = []
    y_train_attack = []
    x_test_attack = []
    y_test_attack = []
    for i in range(10):
        for j in range(100):
            read_path = data_path + f"/{i}/{j}.jpg"
            x_temp = img.imread(read_path)
            # print(x_temp)
            if j < 80:
                x_train_attack.append(x_temp)
                y_train_attack.append(i)
            else:
                x_test_attack.append(x_temp)
                y_test_attack.append(i)
    x_train_attack = np.array(x_train_attack)
    y_train_attack = np.array(y_train_attack)
    x_test_attack = np.array(x_test_attack)
    y_test_attack = np.array(y_test_attack)
    return x_train_attack, y_train_attack, x_test_attack, y_test_attack


def load_test_data(data_path):
    print("load data...")
    x_train_attack = []
    y_train_attack = []
    x_test_attack = []
    y_test_attack = []
    for i in range(10):
        for j in range(100):
            read_path = data_path + f"/{i}/{j}.jpg"
            x_temp = img.imread(read_path)
            # print(x_temp)
            if j < 100:
                x_train_attack.append(x_temp)
                y_train_attack.append(i)
            else:
                x_test_attack.append(x_temp)
                y_test_attack.append(i)
    x_train_attack = np.array(x_train_attack)
    y_train_attack = np.array(y_train_attack)
    x_test_attack = np.array(x_test_attack)
    y_test_attack = np.array(y_test_attack)
    return x_train_attack, y_train_attack, x_test_attack, y_test_attack


def attack_train():
    attack_model = create_attack_model()
    # path = "./data_train/list"
    path = f"./k_{u}/result/data_train_4000/list"
    # path = f"D:/workspace/neural_guess/ultimate/result/data_test_3000/list"
    x, y, x_, y_ = load_train_data(path)
    x_train = x.astype('float32')
    x_test = x_.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = np_utils.to_categorical(y, 10)
    y_test = np_utils.to_categorical(y_, 10)
    x_train = x_train.reshape(x_train.shape[0], 10, 320, 1)
    x_test = x_test.reshape(x_test.shape[0], 10, 320, 1)
    attack_model.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(x_test, y_test))
    attack_model.save(f"./k_{u}/attack_model.h5")
    score = attack_model.evaluate(x_test, y_test, verbose=2)
    print(f'Test loss:{score[0]}, Test accuracy:{score[1]}')


def attack_test():
    test_model = load_model(f"./k_{u}/attack_model.h5")
    path = f"./k_{u}/result/data_test/list"
    x, y, x_, y_ = load_test_data(path)
    x_test = x.astype('float32')
    x_test /= 255
    y_test = np_utils.to_categorical(y, 10)
    x_test = x_test.reshape(x_test.shape[0], 10, 320, 1)
    score = test_model.evaluate(x_test, y_test, verbose=2)
    print(f'Test loss:{score[0]}, Test accuracy:{score[1]}')


if __name__ == '__main__':
    # attack_train()
    usr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for u in usr:
        attack_test()
