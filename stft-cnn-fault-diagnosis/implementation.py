import os
import time

import numpy as np
from scipy.io import loadmat
from scipy.signal import stft
from sklearn import preprocessing
from tensorflow.keras import Input, layers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model

from preprocess import prepro


def read_mat(dir_path):
    filenames = os.listdir(dir_path)
    # fault_names = ['48k_Drive_End_B', '48k_Drive_End_IR', '48k_Drive_End_OR', 'normal']
    fault_names = ['B', 'IR', 'OR', 'normal']
    data_dict = dict()
    for name in filenames:
        # 文件路径
        file_path = os.path.join(dir_path, name)
        file = loadmat(file_path)
        file_keys = file.keys()
        for key in file_keys:
            if 'DE' in key:
                for fault in fault_names:
                    if fault in data_dict.keys():
                        data_dict[fault] = np.hstack((data_dict[fault], file[key].ravel()))
                    else:
                        data_dict[fault] = file[key].ravel()
    return data_dict


def train_test_split(data, test_rate=0.2, data_number=1000, data_length=1024):
    train_samples = dict()
    test_samples = dict()

    for key in data.keys():
        slice_data = data[key]
        total_length = len(slice_data)
        end_index = int(total_length * (1 - test_rate))
        samp_number = int(data_number * (1 - test_rate))
        train_list = list()
        test_ist = list()
        for i in range(samp_number):
            random_start = np.random.randint(low=0, high=(end_index - data_length))
            sample = slice_data[random_start:random_start + data_length]
            train_list.append(sample)

        for j in range(data_number - samp_number):
            random_start = np.random.randint(low=end_index, high=(total_length - data_length))
            sample = slice_data[random_start:random_start + data_length]
            test_ist.append(sample)
        train_samples[key] = train_list
        test_samples[key] = test_ist

    return train_samples, test_samples


def one_hot_label(data):
    data_x = list()
    data_y = list()
    label = 0
    for name in data.keys():
        sample_x = data[name]
        data_x.extend(sample_x)
        len_x = len(sample_x)
        data_y.extend([label] * len_x)
        label += 1

    encoder = preprocessing.OneHotEncoder()
    data_y = np.array(data_y).reshape((-1, 1))
    encoder.fit(data_y)
    data_y = encoder.transform(data_y).toarray()
    return np.asarray(data_x), data_y


def data_to_stft(data):
    stft_list = list()
    for sample in data:
        _, _, zxx = stft(sample, nperseg=64, noverlap=34)
        stft_list.append(zxx[0:32, 0:32])
    return np.asarray(stft_list)


# file_dict = read_mat('../data/12k/0HP')
# train_data, test_data = train_test_split(file_dict)
# train_x, train_y = one_hot_label(train_data)
# test_x, test_y = one_hot_label(test_data)

train_x, train_y, valid_x, valid_y, test_x, test_y = prepro(d_path='../data/48k/0HP',
                                                            length=1024,
                                                            number=1000,
                                                            normal=False,
                                                            rate=[0.5, 0.25, 0.25],
                                                            enc=False)

train_x = data_to_stft(train_x)
valid_x = data_to_stft(valid_x)
test_x = data_to_stft(test_x)

train_x = np.expand_dims(train_x, -1)
valid_x = np.expand_dims(valid_x, -1)
test_x = np.expand_dims(test_x, -1)

input_size = train_x.shape[1:]
output_size = train_y.shape[-1]
model = Sequential([
    Input(shape=input_size),
    layers.AveragePooling2D(pool_size=2, strides=2),
    layers.Conv2D(filters=8, kernel_size=3, strides=1, kernel_regularizer=l2(1e-4)),
    layers.AveragePooling2D(pool_size=2, strides=2),
    layers.Conv2D(filters=16, kernel_size=3, strides=1, kernel_regularizer=l2(1e-4)),
    layers.Flatten(),
    layers.Dense(units=400, activation='relu'),
    layers.Dense(units=output_size, activation='softmax'),
])
model.summary()

opt = Adam(learning_rate=0.05)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
localtime = time.strftime("%Y%m%d_%H%M", time.localtime())
tb_cb = TensorBoard(log_dir=rf'logs\stft_cnn_fault_diagnosis-{localtime}')
model.fit(x=train_x, y=train_y, batch_size=64, epochs=30,
          validation_data=(valid_x, valid_y),
          verbose=1, shuffle=True, callbacks=[tb_cb])

score = model.evaluate(x=test_x, y=test_y)
print("测试集上的损失率：", score[0])
print("测试集上的准确率：", score[1])
plot_model(model=model, to_file='images/stft_cnn_fault_diagnosis.png', show_shapes=True)
