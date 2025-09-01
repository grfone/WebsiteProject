import tensorflow as tf
import tensorflow_probability as tfp


class MixBatch:
    """A class to apply Mixup and CutMix augmentations to a batch of images and labels."""

    def __init__(self):
        pass

    def apply_mixup_cutmix(self, images, labels):
        """Randomly apply Mixup or CutMix augmentation."""
        choice = tf.random.uniform([], 0, 1)
        return tf.cond(choice < 0.5,
                       lambda: self._mixup(images, labels),
                       lambda: self._cutmix(images, labels))

    @staticmethod
    def _sample_beta(alpha):
        """Sample a value from a Beta distribution using TensorFlow ops."""
        beta_dist = tfp.distributions.Beta(alpha, alpha)
        return beta_dist.sample()

    def _mixup(self, images, labels, alpha=0.2):
        """Apply Mixup augmentation."""
        batch_size = tf.shape(images)[0]
        lam = self._sample_beta(alpha)
        index = tf.random.shuffle(tf.range(batch_size))
        mixed_images = lam * images + (1 - lam) * tf.gather(images, index)
        mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, index)
        return mixed_images, mixed_labels

    def _cutmix(self, images, labels, alpha=1.0):
        """Apply CutMix augmentation."""
        batch_size = tf.shape(images)[0]
        W, H = tf.shape(images)[1], tf.shape(images)[2]

        lam = self._sample_beta(alpha)
        r_x = tf.random.uniform([], 0, W, dtype=tf.int32)
        r_y = tf.random.uniform([], 0, H, dtype=tf.int32)
        r_w = tf.cast(tf.cast(W, tf.float32) * tf.sqrt(1.0 - lam), tf.int32)
        r_h = tf.cast(tf.cast(H, tf.float32) * tf.sqrt(1.0 - lam), tf.int32)

        x1 = tf.clip_by_value(r_x - r_w // 2, 0, W)
        y1 = tf.clip_by_value(r_y - r_h // 2, 0, H)
        x2 = tf.clip_by_value(r_x + r_w // 2, 0, W)
        y2 = tf.clip_by_value(r_y + r_h // 2, 0, H)

        index = tf.random.shuffle(tf.range(batch_size))
        shuffled_images = tf.gather(images, index)
        shuffled_labels = tf.gather(labels, index)

        # Create mask for CutMix
        mask = tf.pad(
            tf.ones((y2 - y1, x2 - x1, 3)),
            [[y1, H - y2], [x1, W - x2], [0, 0]],
            constant_values=0
        )

        mask = tf.expand_dims(mask, 0)
        images = images * (1 - mask) + shuffled_images * mask

        lam_adjusted = 1 - tf.cast((x2 - x1) * (y2 - y1), tf.float32) / tf.cast(W * H, tf.float32)
        labels = lam_adjusted * labels + (1 - lam_adjusted) * shuffled_labels
        return images, labels
