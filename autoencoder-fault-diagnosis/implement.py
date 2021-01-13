import time

from tensorflow.keras import Input, layers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from preprocess import prepro

train_x, train_y, valid_x, valid_y, test_x, test_y = prepro(d_path='../data/48k/0HP',
                                                            length=400,
                                                            number=400,
                                                            normal=True,
                                                            rate=[0.5, 0.25, 0.25],
                                                            enc=False)

# train_x = np.expand_dims(train_x, -1)
# valid_x = np.expand_dims(valid_x, -1)
# test_x = np.expand_dims(test_x, -1)

input_size = train_x.shape[1:]
output_size = train_y.shape[-1]

input_layer = Input(shape=input_size)
encoded = layers.Dense(300, activation='relu')(input_layer)
encoded = layers.Dropout(0.5)(encoded)
encoded = layers.Dense(220, activation='relu')(encoded)
encoded = layers.Dropout(0.5)(encoded)
encoded = layers.Dense(90, activation='relu')(encoded)
# encoded = layers.Dropout(0.5)(encoded)
# encoded = layers.Dense(10, activation='softmax')(encoded)

# decoded = layers.Dense(90, activation='relu')(encoded)
# decoded = layers.Dropout(0.5)(decoded)
decoded = layers.Dense(220, activation='relu')(encoded)
decoded = layers.Dropout(0.5)(decoded)
decoded = layers.Dense(300, activation='relu')(decoded)
decoded = layers.Dropout(0.5)(decoded)
decoded = layers.Dense(400, activation='tanh')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='Adam', loss='mean_squared_error')

localtime = time.strftime("%Y%m%d_%H%M", time.localtime())
tb_cb = TensorBoard(log_dir=rf'logs\autoencoder-{localtime}')
autoencoder.fit(train_x, train_x,
                epochs=500,
                batch_size=256,
                shuffle=True,
                validation_data=(valid_x, valid_x),
                callbacks=[tb_cb])

# encoder = Model(autoencoder.input, autoencoder.layers[7].output, trainable=False)
# model_weights = autoencoder.get_weights()
# autoencoder.load_weights(model_weights[:6])
encoder = Sequential(autoencoder.layers[:6])
# encoder.trainable = False
encoder.add(layers.Dense(10, activation='softmax'))
encoder.summary()

# opt = Adam(learning_rate=1e-5)
encoder.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

tb_cb = TensorBoard(log_dir=rf'logs\encoder-{localtime}')
encoder.fit(train_x, train_y,
                epochs=200,
                batch_size=256,
                shuffle=True,
                validation_data=(valid_x, valid_y),
                callbacks=[tb_cb])

score = encoder.evaluate(x=test_x, y=test_y)
print("测试集上的损失率：", score[0])
print("测试集上的准确率：", score[1])
plot_model(model=autoencoder, to_file='images/autoencoder.png', show_shapes=True)
