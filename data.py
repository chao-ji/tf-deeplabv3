"""Data module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

B_MEAN, G_MEAN, R_MEAN = 104.00698793, 116.66876762, 122.67891434
VOID_LABEL = 255
AUGTRAIN_SIZE = 10582
TRAIN_MODE = tf.contrib.learn.ModeKeys.TRAIN
INFER_MODE = tf.contrib.learn.ModeKeys.INFER


class PascalVOCDataset(object):
  """"""
  def __init__(self, hparams, mode, scope=None):
    self._mode = mode
    self._batch_size = hparams.batch_size if mode == TRAIN_MODE else 1

    with tf.variable_scope(scope, "data"):
      if mode != INFER_MODE:
        self._labels, self._images = self._build_dataset(hparams)
#    else:
#      (self._initializer, self._labels, self._mask, self._num_valid_pixels,
#          self._images) = self._get_infer_iterator(hparams)

  @property
  def mode(self):
    return self._mode

  @property
  def labels(self):
    return self._labels

  @property
  def images(self):
    return self._images

  def _build_dataset(self, hparams):
    """Builds the dataset pipeline for training or evaluation."""
    labels_dir = hparams.labels_dir
    images_dir = hparams.images_dir
    image_sets_dir = hparams.image_sets_dir
    random_seed = hparams.random_seed
    batch_size = self._batch_size

    if self.mode == TRAIN_MODE:
      fid = os.path.join(image_sets_dir, "augtrain.txt") 
    else:
      fid = os.path.join(image_sets_dir, "val.txt")

    label_fns = [os.path.join(labels_dir, (l.strip() + ".png"))
        for l in tf.gfile.GFile(fid).readlines()]
    image_fns = [os.path.join(images_dir, (l.strip() + ".jpg"))
        for l in tf.gfile.GFile(fid).readlines()]

    dataset = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(label_fns)),
        tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(image_fns))))

    if self.mode == TRAIN_MODE:
      dataset = dataset.repeat().shuffle(AUGTRAIN_SIZE, random_seed)

    dataset = dataset.map(lambda fn_lbl, fn_img: 
        (tf.read_file(fn_lbl), tf.read_file(fn_img)))

    dataset = dataset.map(lambda str_lbl, str_img:
        (tf.image.decode_png(str_lbl, channels=1),
        tf.image.decode_jpeg(str_img, channels=3)))

    if self.mode == TRAIN_MODE:
      dataset = dataset.map(
          lambda lbl, img: _augment_training_data(lbl, img, hparams))

    dataset = dataset.map(lambda lbl, img: (lbl, _subtract_channel_means(img)))

    dataset = dataset.map(lambda lbl, img:
        (tf.squeeze(tf.cast(lbl, tf.int32), axis=2), img))

#    def get_masks(lbl):
#      return tf.where(tf.equal(lbl, VOID_LABEL),
#          tf.zeros_like(lbl, dtype=tf.float32),
#          tf.ones_like(lbl, dtype=tf.float32))

#    def zero_out_void(lbl):
#      return tf.where(tf.equal(lbl, VOID_LABEL),
#        tf.zeros_like(lbl, dtype=tf.int32), lbl)

#    dataset = dataset.map(
#        lambda lbl, img: (lbl, img, get_masks(lbl))).map(
#        lambda lbl, img, mask: (zero_out_void(lbl), img, mask)).map(
#        lambda lbl, img, mask: (lbl, img, mask, tf.reduce_sum(mask)))

    dataset = dataset.batch(batch_size).prefetch(buffer_size=batch_size * 2)

    labels, images = dataset.make_one_shot_iterator().get_next()

    return labels, images


def _augment_training_data(label, image, hparams):
  """Augment training data by random rescaling, cropping and horizontal 
  flipping."""
  rescale_range = hparams.rescale_range
  crop_image_size = hparams.crop_image_size
  seed = hparams.random_seed

  label = tf.cast(label, tf.float32)
  image = tf.cast(image, tf.float32)

  label, image = _random_rescale(label, image, rescale_range, seed)
  label, image = _random_crop_and_flip(label, image, crop_image_size, seed)

  return label, image 


def _random_rescale(label, image, rescale_range, seed=None):
  """Randomly rescale labels and images."""
  min_scale, max_scale = rescale_range

  scale = tf.random_uniform([],
       minval=min_scale, maxval=max_scale, dtype=tf.float32, seed=seed)
  size = tf.cast(tf.shape(label), tf.float32)
  new_size = tf.cast(size[:2] * scale, tf.int32)
  
  label = tf.image.resize_images(
      label, new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  image = tf.image.resize_images(
      image, new_size, method=tf.image.ResizeMethod.BILINEAR)

  return label, image


def _random_crop_and_flip(label, image, crop_image_size, seed=None):
  """Randomly crop (possibly w/ padding) labels and images to a given spatial 
  size, then do a horizontal flip.
  """
  crop_height, crop_width = crop_image_size

  # Subtracting `VOID_LABEL` first and adding it back is a trick to effectively
  # pad `VOID_LABEL` values instead of the default padding value 
  # (i.e. zeros) of `tf.image.pad_to_bounding_box`
  label = label - VOID_LABEL

  shape = tf.shape(label)
  height, width = shape[0], shape[1]
  stacked = tf.concat([label, image], axis=2)

  target_height= tf.maximum(crop_height, height)
  target_width = tf.maximum(crop_width, width)
  offset_height = (target_height - height) // 2
  offset_width = (target_width - width) // 2

  stacked = tf.image.pad_to_bounding_box(
      stacked, offset_height, offset_width, target_height, target_width)
  stacked = tf.random_crop(stacked, [crop_height, crop_width, 4], seed)
  stacked = tf.image.random_flip_left_right(stacked, seed)
      
  label, image = tf.split(stacked, [1, 3], axis=2)
  label += VOID_LABEL

  return label, image


def _subtract_channel_means(image):
  image.set_shape([None, None, 3])
  if image.dtype != tf.float32:
    image = tf.cast(image, tf.float32)
  r, g, b = tf.unstack(image, axis=2)
  image = tf.stack([r - R_MEAN, g - G_MEAN, b - B_MEAN], axis=2)
  return image