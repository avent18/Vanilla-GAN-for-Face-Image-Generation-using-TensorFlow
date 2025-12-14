import tensorflow as tf
from tensorflow.keras import layers
import os

class CelebADatasetTF:
    def __init__(self, root_dir, image_size=(128, 128), batch_size=32, shuffle=True):
        self.root_dir = root_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.image_paths = [
            os.path.join(root_dir, img)
            for img in os.listdir(root_dir)
            if img.endswith(".jpg")
        ]

    def _load_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.image_size)
        image = tf.cast(image, tf.float32) / 255.0  # normalize [0,1]
        return image

    def get_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.image_paths)
        dataset = dataset.map(
            lambda x: self._load_image(x),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
