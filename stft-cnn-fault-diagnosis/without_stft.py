import time

import numpy as np
from tensorflow.keras import Input, layers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model

from implementation import read_mat
from preprocess import prepro

file_dict = read_mat('../data/12k/0HP')
# train_data, test_data = train_test_split(file_dict)
# train_x, train_y = one_hot_label(train_data)
# test_x, test_y = one_hot_label(test_data)

train_x, train_y, valid_x, valid_y, test_x, test_y = prepro(d_path='../data/48k/0HP',
                                                            length=1024,
                                                            number=1000,
                                                            normal=False,
                                                            rate=[0.5, 0.25, 0.25],
                                                            enc=False)

train_x = np.expand_dims(train_x, -1)
valid_x = np.expand_dims(valid_x, -1)
test_x = np.expand_dims(test_x, -1)

input_size = train_x.shape[1:]
output_size = train_y.shape[-1]
model = Sequential([
    Input(shape=input_size),
    layers.AveragePooling1D(pool_size=2, strides=2),
    layers.Conv1D(filters=8, kernel_size=3, strides=1, kernel_regularizer=l2(1e-4)),
    layers.AveragePooling1D(pool_size=2, strides=2),
    layers.Conv1D(filters=16, kernel_size=3, strides=1, kernel_regularizer=l2(1e-4)),
    layers.Flatten(),
    layers.Dense(units=400, activation='relu'),
    layers.Dense(units=output_size, activation='softmax', kernel_regularizer=l2(1e-4)),
])
model.summary()

opt = Adam(learning_rate=0.05)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
localtime = time.strftime("%Y%m%d_%H%M", time.localtime())
tb_cb = TensorBoard(log_dir=rf'logs\without_stft_cnn_fault_diagnosis-{localtime}')
model.fit(x=train_x, y=train_y, batch_size=128, epochs=30,
          validation_data=(valid_x, valid_y),
          verbose=1, shuffle=True, callbacks=[tb_cb])

score = model.evaluate(x=test_x, y=test_y)
print("测试集上的损失率：", score[0])
print("测试集上的准确率：", score[1])
plot_model(model=model, to_file='images/without_stft_cnn_fault_diagnosis.png', show_shapes=True)
