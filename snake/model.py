from PIL import Image  # To transform the image in the Processor
import numpy as np
from rl.core import Processor
import tensorflow as tf

# Convolutional Backbone Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution2D, Permute

class ImageProcessor(Processor):
    def __init__(self, img_shape):
        super().__init__()
        self.img_shape = img_shape

    def process_observation(self, observation):
        # First convert the numpy array to a PIL Image
        img = Image.fromarray(observation)
        # Then resize the image
        img = img.resize(self.img_shape)
        # And convert it to grayscale  (The L stands for luminance)
        img = img.convert("L")
        # Convert the image back to a numpy array and finally return the image
        img = np.array(img)
        return img.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        return batch.astype('float32') / 255.

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


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

class BaseModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.sequential = Sequential()
    def call(self, inputs, training=False):
        return self.sequential(inputs)


class SnakeModel(tf.keras.Model):
    def __init__(self, input_shape, nb_actions):
        super().__init__()
        self.sequential = Sequential()
        self.sequential.add(Permute((2, 3, 1), input_shape=input_shape))
        self.sequential.add(Convolution2D(32, (8, 8), strides=(
                4, 4), kernel_initializer='he_normal'))
        self.sequential.add(Activation('relu'))
        self.sequential.add(Convolution2D(64, (4, 4), strides=(
                2, 2), kernel_initializer='he_normal'))
        self.sequential.add(Activation('relu'))
        self.sequential.add(Convolution2D(64, (3, 3), strides=(
                1, 1), kernel_initializer='he_normal'))
        self.sequential.add(Activation('relu'))
        self.sequential.add(Flatten())
        self.sequential.add(Dense(512))
        self.sequential.add(Activation('relu'))
        self.sequential.add(Dense(nb_actions))
        self.sequential.add(Activation('linear'))

    def call(self, inputs, training=False):
        return self.sequential(inputs)


        # Permute((2, 3, 1), input_shape= input_shape),
        #     Convolution2D(32, (8, 8), strides = (4, 4), kernel_initializer = 'he_normal'),
        #     Activation('relu'),
        #     Convolution2D(64, (4, 4), strides = (2, 2), kernel_initializer = 'he_normal'),
        #     Activation('relu'),
        #     Convolution2D(64, (3, 3), strides = (1, 1), kernel_initializer = 'he_normal'),
        #     Activation('relu'),
        #     Flatten(),
        #     Dense(512),
        #     Activation('relu'),
        #     Dense(nb_actions),
        #     Activation('linear')
