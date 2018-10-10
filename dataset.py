import tensorflow as tf

RGB_MEAN = 122.67891434, 116.66876762, 104.00698793 


class BaseTrainerEvaluatorDataset(object):
  """Base class to be subclassed by Trainer and Evaluator dataset.

  Implemented method `_decode_raw_protobuf_string` to convert raw protocol
  buffer string (scalar tensor) into images and labels.
  """
  def _decode_raw_protobuf_string(self, protobuf_string):
    """Decodes raw proto buffer string scalar tensor into a tensor dict.

    Args:
      protobuf_string: string scalar tensor, protobuf string for one example.

    Returns:
      tensor_dict: dict mapping from tensor name to tensors (3-D float tensors
        of shape [height, width, channels]).
    """
    keys_to_features = _get_keys_to_features()
    tensor_dict = tf.parse_single_example(protobuf_string, keys_to_features)
    return {'images': tf.image.decode_jpeg(tensor_dict['image'], channels=3),
            'labels': tf.image.decode_png(tensor_dict['label'], channels=1)}


class TrainerDeepLabV3Dataset(BaseTrainerEvaluatorDataset):
  """Dataset for training a DeepLabV3 model."""
  def __init__(self, 
               batch_size=12,
               pad_imagenet_mean=False,
               min_rescale_factor=0.5, 
               max_rescale_factor=2.0, 
               rescale_step_size=None, 
               crop_height=513, 
               crop_width=513, 
               ignore_label=255,
               shuffle_buffer_size=10000):
    """Constructor.

    Args:
      batch_size: int scalar, batch size.
      pad_imagenet_mean: bool scalar, whether to pad the images with
        imagenet mean (True) or `[127.5, 127.5, 127.5]` (False).
      min_rescale_factor: float scalar, lower bound of image rescaling factor.
      max_rescale_factor: float scalar, upper bound of image rescaling factor.
      rescale_step_size: float scalar or None, rescale factor is sampled from
        list `range(min_rescale_factor, max_rescale_factor, rescale_step_size)`.
        If None, it is instead sampled by 
        `tf.random_uniform([], min_rescale_factor, max_rescale_factor)`.
      crop_height: int scalar, height of cropped image.
      crop_width: int scalar, width of cropped image.
      ignore_label: int scalar, integer representing the class in `labels` to 
        be ignored (i.e. masked out when computing loss).
      shuffle_buffer_size: int scalar, shuffle buffer size.
    """
    self._batch_size = batch_size
    self._pad_imagenet_mean = pad_imagenet_mean
    self._min_rescale_factor = min_rescale_factor
    self._max_rescale_factor = max_rescale_factor
    self._rescale_step_size = rescale_step_size
    self._crop_height = crop_height
    self._crop_width = crop_width
    self._ignore_label = ignore_label
    self._shuffle_buffer_size = shuffle_buffer_size

  @property
  def mode(self):
    return tf.contrib.learn.ModeKeys.TRAIN

  def get_tensor_dict(self, filenames):
    """Generates tensor dict for training.

    Args:
      filenames: list of strings, the list of TFRecord filenames.

    Returns:
      tensor_dict: a dict mapping from tensor names to tensors:
        { 'images': [batch_size, height, width, channels=3], tf.float32
          'labels': [batch_size, height, width], tf.int32 }
    """
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.repeat().shuffle(self._shuffle_buffer_size)

    dataset = dataset.map(lambda protobuf_string: 
        self._decode_raw_protobuf_string(protobuf_string))
    dataset = dataset.map(lambda tensor_dict: 
        self._do_augmentation(tensor_dict))

    dataset = dataset.batch(self._batch_size, drop_remainder=True)
    tensor_dict = dataset.make_one_shot_iterator().get_next()

    return tensor_dict

  def _do_augmentation(self, tensor_dict):
    """Performs data augmentation on images and labels.

    Args:
      tensor_dict: a dict mapping from tensor names to tensors:
        { 'images': [height, width, channels=3]
          'labels': [height, width, channels=1] }

    Returns:
      tensor_dict: a dict mapping from tensor names to tensors:
        { 'images': [height, width, channels=3]
          'labels': [height, width, channels=1] }
    """
    images = tf.to_float(tensor_dict['images'])
    labels = tf.to_float(tensor_dict['labels'])
    images, labels = self._random_rescale(images, labels)
    images, labels = self._pad_randomcrop_randomflip(images, labels)
    tensor_dict['images'] = images
    tensor_dict['labels'] = tf.squeeze(tf.to_int32(labels), axis=2)
    return tensor_dict

  def _random_rescale(self, images, labels):
    """Randomly rescales input images and labels."""
    if self._rescale_step_size is None:
      scale = tf.random_uniform([],
          minval=self._min_rescale_factor, maxval=self._max_rescale_factor)
    else:
      num_steps = int((self._max_rescale_factor - self._min_rescale_factor) / 
          self._rescale_step_size + 1)
      rescale_factors = tf.lin_space(
          self._min_rescale_factor, self._max_rescale_factor, num_steps)
      index = tf.random_uniform([], 
          minval=0, maxval=tf.size(rescale_factors), dtype=tf.int32)
      scale = rescale_factors[index]

    new_size = tf.to_int32(tf.to_float(tf.shape(labels)[:2]) * scale)

    images = tf.image.resize_images(images, new_size, 
        method=tf.image.ResizeMethod.BILINEAR)
    labels = tf.image.resize_images(labels, new_size, 
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return images, labels

  def _pad_randomcrop_randomflip(self, images, labels):
    """Pad `images` and `labels` to spatial dimension no smaller than 
    `[crop_height, crop_width]`, then randomly crop a patch of this size
    and flip with probability 1/2.
    """
    height, width = tf.unstack(tf.shape(images)[:2])

    target_height= tf.maximum(self._crop_height, height)
    target_width = tf.maximum(self._crop_width, width)

    offset_height = (target_height - height) // 2
    offset_width = (target_width - width) // 2
 
    image_pad_value = self._get_pad_value()
    images -= image_pad_value
    labels -= self._ignore_label
    # `images`, `labels` are lumped so they can be cropped and flipped together.
    images_labels = tf.concat([images, labels], axis=2)

    images_labels = tf.image.pad_to_bounding_box(
        images_labels, offset_height, offset_width, target_height, target_width)

    images_labels = tf.random_crop(
        images_labels, [self._crop_height, self._crop_width, 4])
    images_labels = tf.image.random_flip_left_right(images_labels)
    images_labels.set_shape([self._crop_height, self._crop_width, 4])

    images, labels = tf.split(images_labels, [3, 1], axis=2)
    images += image_pad_value
    labels += self._ignore_label

    return images, labels

  def _get_pad_value(self):
    """Returns the pixel value (tensor holding R,G,B values) to pad images."""
    return tf.reshape(tf.convert_to_tensor(RGB_MEAN if self._pad_imagenet_mean 
        else (127.5, 127.5, 127.5)), [1, 1, 3])


class EvaluatorDeepLabV3Dataset(BaseTrainerEvaluatorDataset):
  """Dataset for evaluating a DeepLabV3 model."""

  @property
  def mode(self):
    return tf.contrib.learn.ModeKeys.EVAL

  def get_tensor_dict(self, filenames):
    """Generates tensor dict for evaluation.

    Args:
      filenames: list of strings, the list of TFRecord filenames.

    Returns:
      tensor_dict: a dict mapping from tensor names to tensors:
        { 'images': [1, height, width, channels=3], tf.float32
          'labels': [1, height, width], tf.int32 }
    """
    dataset = tf.data.TFRecordDataset(filenames)

    dataset = dataset.map(lambda protobuf_string:
        self._decode_raw_protobuf_string(protobuf_string))

    dataset = dataset.map(lambda tensor_dict: {
        'images': tf.to_float(tensor_dict['images']), 
        'labels': tf.squeeze(tf.to_int32(tensor_dict['labels']), axis=2)})
    dataset = dataset.batch(1)
    tensor_dict = dataset.make_one_shot_iterator().get_next()
    tensor_dict['images'].set_shape([1, None, None, 3])
    tensor_dict['labels'].set_shape([1, None, None])
    return tensor_dict
    

class InferencerDeepLabV3Dataset(object):
  """Dataset for making inference using a DeepLabV3 model."""

  @property
  def mode(self):
    return tf.contrib.learn.ModeKeys.INFER

  def get_tensor_dict(self, filenames):
    """Generates tensor dict for making inference.

    Args:
      filenames: list of strings, the list of raw jpeg image files.

    Returns:
      tensor_dict: a dict mapping from tensor names to tensors:
        { 'images': [1, height, width, channels=3], tf.float32,
          'filename': filename of input image, string scalar}
    """
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    dataset = dataset.map(lambda filename: (
        tf.to_float(tf.image.decode_jpeg(tf.read_file(filename), channels=3)), 
        filename))

    images, filename = dataset.make_one_shot_iterator().get_next()
    images.set_shape([None, None, 3])
    tensor_dict = {'images': tf.expand_dims(images, 0), 'filename': filename}
    return tensor_dict


def _get_keys_to_features():
  """Returns a dict mapping from field name of a protobuf example to the 
  corresponding parser.
  """
  keys_to_features = {
      'image': tf.FixedLenFeature((), tf.string, default_value=''),
      'label': tf.FixedLenFeature((), tf.string, default_value='')}
  return keys_to_features

