import tensorflow as tf
import numpy as np
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras.applications import efficientnet
from src.TypesOfClouds_extra.CustomDataManager_extra.MixBatch import MixBatch


class CustomDataManager:

    def make_dataset(self, paths, labels, is_train_set):
        """Create a TensorFlow dataset from image paths and labels.

        Args:
            paths (list or np.ndarray): List of file paths to images.
            labels (list or np.ndarray): Corresponding labels for the images.
            is_train_set (bool): Whether the dataset is for training (applies augmentations).

        Returns:
            tf.data.Dataset: A preprocessed and batched dataset.
        """
        # Convert inputs to tensors
        paths = tf.convert_to_tensor(paths, dtype=tf.string)
        labels = tf.convert_to_tensor(labels)

        # Create dataset from tensors
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        if is_train_set:
            ds = ds.shuffle(buffer_size=min(len(paths), 4000), seed=42, reshuffle_each_iteration=True)

        # Map parsing function with is_train_set passed explicitly
        ds = ds.map(lambda p, y: self._parse(p, y, is_train_set), num_parallel_calls=AUTOTUNE).batch(16)

        if is_train_set:
            ds = ds.map(MixBatch().apply_mixup_cutmix, num_parallel_calls=AUTOTUNE)

        return ds.prefetch(AUTOTUNE)

    def _parse(self, path, label, is_train_set):
        """Parse and preprocess an image from its file path.

        Args:
            path (tf.Tensor): File path to the image.
            label (tf.Tensor): Corresponding label.
            is_train_set (bool): Whether to apply training augmentations.

        Returns:
            tuple: Preprocessed image and label.
        """
        # Read and decode image
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)

        if is_train_set:
            img = tf.image.random_crop(img, size=[380, 380, 3])
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.1)
            img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
            img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
            img = tf.image.random_hue(img, max_delta=0.05)
            img = tf.image.random_jpeg_quality(img, min_jpeg_quality=75, max_jpeg_quality=100)
        else:
            img = tf.image.resize(img, [380, 380], method="bicubic")

        img = efficientnet.preprocess_input(img * 255.0)
        return img, label
