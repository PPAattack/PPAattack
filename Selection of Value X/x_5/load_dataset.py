from tensorflow.keras.datasets import cifar10
import numpy as np


def split_dataset():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train_sort = []
    y_train_sort = []
    x_test_sort = []
    y_test_sort = []
    for i in range(10):
        x_tr = []
        y_tr = []
        x_te = []
        y_te = []
        for j in range(len(x_train)):
            if y_train[j] == i:
                x_tr.append(x_train[j])
                y_tr.append(y_train[j])
            else:
                continue
        x_train_sort.append(x_tr)
        y_train_sort.append(y_tr)
        for k in range(len(x_test)):
            if y_test[k] == i:
                x_te.append(x_test[k])
                y_te.append(y_test[k])
            else:
                continue
        x_test_sort.append(x_te)
        y_test_sort.append(y_te)
    # for item_x, item_y in zip(x_train_sort, y_train_sort):
    #     print("---------------train---------------")
    #     print(f"x:{len(item_x)}, y:{len(item_y)}")
    # for item_x, item_y in zip(x_test_sort, y_test_sort):
    #     print("---------------test---------------")
    #     print(f"x:{len(item_x)}, y:{len(item_y)}")
    return x_train_sort, y_train_sort, x_test_sort, y_test_sort


def get_server_dataset(num=200):
    x_train, y_train, x_test, y_test = split_dataset()
    x_server = []
    y_server = []
    for i in range(10):
        for j in range(num):
            x_server.append(x_train[i][j])
            y_server.append(y_train[i][j])
    x_server = np.array(x_server)
    y_server = np.array(y_server)
    # print(f"x:{x_server.shape}, y:{y_server.shape}")
    return x_server, y_server, x_server, y_server


def get_server_dataset_expend(num=200):
    x_train, y_train, x_test, y_test = split_dataset()
    x_server = []
    y_server = []
    for i in range(10):
        for j in range(num * 5):
            x_server.append(x_train[i][j % num])
            y_server.append(y_train[i][j % num])
    x_server = np.array(x_server)
    y_server = np.array(y_server)
    # print(f"x:{x_server.shape}, y:{y_server.shape}")
    return x_server, y_server


class client:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


def get_client_dataset(server_num, num):
    x_train, y_train, x_test, y_test = split_dataset()
    client_list = []
    for c in range(10):
        x_client = []
        y_client = []
        for i in range(10):
            if i == c:
                for j in range(num):
                    x_client.append(x_train[i][server_num + j])
                    y_client.append(y_train[i][server_num + j])
            else:
                for j in range(200):
                    x_client.append(x_train[i][server_num + j])
                    y_client.append(y_train[i][server_num + j])
        x_client = np.array(x_client)
        y_client = np.array(y_client)

        x_client_test = []
        y_client_test = []
        for i in range(10):
            for j in range(80):
                x_client_test.append(x_test[i][80 * c + j])
                y_client_test.append(y_test[i][80 * c + j])
        x_client_test = np.array(x_client_test)
        y_client_test = np.array(y_client_test)

        cl = client(x_client, y_client, x_client_test, y_client_test)
        client_list.append(cl)
    return client_list


def get_client_dataset_part(server_num, num, k):
    x_train, y_train, x_test, y_test = split_dataset()
    client_list = []
    for n in range(6):
        x_client = []
        y_client = []
        if n == 0:
            for i in range(10):
                if i == k:
                    for j in range(num):
                        x_client.append(x_train[i][server_num + j])
                        y_client.append(y_train[i][server_num + j])
                else:
                    for j in range(200):
                        x_client.append(x_train[i][server_num + j])
                        y_client.append(y_train[i][server_num + j])
        else:
            for i in range(10):
                if i == k:
                    continue
                else:
                    for j in range(500):
                        x_client.append(x_train[i][server_num + j])
                        y_client.append(y_train[i][server_num + j])
        x_client = np.array(x_client)
        y_client = np.array(y_client)

        x_client_test = []
        y_client_test = []
        for i in range(10):
            for j in range(80):
                x_client_test.append(x_test[i][80 * n + j])
                y_client_test.append(y_test[i][80 * n + j])
        x_client_test = np.array(x_client_test)
        y_client_test = np.array(y_client_test)

        cl = client(x_client, y_client, x_client_test, y_client_test)
        client_list.append(cl)
    return client_list


def get_client_dataset_fake():
    x_train, y_train, _, _ = get_server_dataset(200)
    client_list_fake = []
    for s in range(10):
        x_client_fake = []
        y_client_fake = []
        for i in range(10):
            if i == s:
                for j in range(3000):
                    x_client_fake.append(x_train[i * 200 + j % 200])
                    y_client_fake.append(y_train[i * 200 + j % 200])
            else:
                for j in range(200):
                    x_client_fake.append(x_train[i * 200 + j % 200])
                    y_client_fake.append(y_train[i * 200 + j % 200])
        x_client_fake = np.array(x_client_fake)
        y_client_fake = np.array(y_client_fake)
        cl_fake = client(x_client_fake, y_client_fake, x_client_fake, y_client_fake)
        client_list_fake.append(cl_fake)
    return client_list_fake


def get_client_dataset_part_fake(k):
    x_train, y_train, _, _ = get_server_dataset(200)
    client_list_fake = []
    for n in range(6):
        x_client_fake = []
        y_client_fake = []
        if n == 0:
            for i in range(10):
                if i == k:
                    for j in range(3000):
                        x_client_fake.append(x_train[i * 200 + j % 200])
                        y_client_fake.append(y_train[i * 200 + j % 200])
                else:
                    for j in range(200):
                        x_client_fake.append(x_train[i * 200 + j % 200])
                        y_client_fake.append(y_train[i * 200 + j % 200])
        else:
            for i in range(10):
                if i == k:
                    continue
                else:
                    for j in range(500):
                        x_client_fake.append(x_train[i * 200 + j % 200])
                        y_client_fake.append(y_train[i * 200 + j % 200])
        x_client_fake = np.array(x_client_fake)
        y_client_fake = np.array(y_client_fake)
        cl_fake = client(x_client_fake, y_client_fake, x_client_fake, y_client_fake)
        client_list_fake.append(cl_fake)
    return client_list_fake


def get_client_dataset_all(server_num, num, k):
    x_train, y_train, x_test, y_test = split_dataset()
    client_list = []
    for n in range(10):
        x_client = []
        y_client = []
        if n == 0:
            for i in range(10):
                if i == k:
                    for j in range(num):
                        x_client.append(x_train[i][server_num + j])
                        y_client.append(y_train[i][server_num + j])
                else:
                    for j in range(200):
                        x_client.append(x_train[i][server_num + j])
                        y_client.append(y_train[i][server_num + j])
        elif n == 1 or n == 2:
            for i in range(10):
                if i == k:
                    continue
                else:
                    for j in range(500):
                        x_client.append(x_train[i][server_num + j + n * 500])
                        y_client.append(y_train[i][server_num + j + n * 500])
        else:
            for i in range(10):
                for j in range(500):
                    x_client.append(x_train[i][server_num + (j + n * 500) % 4000])
                    y_client.append(y_train[i][server_num + (j + n * 500) % 4000])
        x_client = np.array(x_client)
        y_client = np.array(y_client)

        x_client_test = []
        y_client_test = []
        for i in range(10):
            for j in range(80):
                x_client_test.append(x_test[i][80 * n + j])
                y_client_test.append(y_test[i][80 * n + j])
        x_client_test = np.array(x_client_test)
        y_client_test = np.array(y_client_test)

        cl = client(x_client, y_client, x_client_test, y_client_test)
        client_list.append(cl)
    return client_list


if __name__ == '__main__':
    split_dataset()

    # x_train_server, y_train_server, _, _ = get_server_dataset(200)

    # x_train_server, y_train_server = get_server_dataset_expend(200)

    # user = get_client_dataset(200, 3000)
    # for item in user:
    #     print(f"x_train:{item.x_train.shape}, x_test:{item.x_test.shape}")

    # user_fake = get_client_dataset_fake()
    # for item in user_fake:
    #     print(f"x_train:{item.x_train.shape}, x_test:{item.x_test.shape}")

    # for m in range(10):
    #     user_p = get_client_dataset_part(200, 3000, m)
    #     for item in user_p:
    #         print(f"x_train:{item.x_train.shape}, x_test:{item.x_test.shape}")

    # for m in range(10):
    #     user_p_fake = get_client_dataset_part_fake(m)
    #     for item in user_p_fake:
    #         print(f"x_train:{item.x_train.shape}, x_test:{item.x_test.shape}")
