from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Permute


def get_model(input_shape, nb_actions):
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), kernel_initializer='he_normal', activation='relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), kernel_initializer='he_normal', activation='relu')),
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), kernel_initializer='he_normal', activation='relu')),
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model