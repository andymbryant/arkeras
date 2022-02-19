from PIL import Image  # To transform the image in the Processor
import numpy as np
from rl.core import Processor

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
