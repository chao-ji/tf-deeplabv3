import tensorflow as tf
import numpy as np
from tensorflow import logging

from nets.resnet_v1 import resnet_v1_50
from nets.resnet_v1 import resnet_v1_101
from nets.resnet_v2 import resnet_v2_50
from nets.resnet_v2 import resnet_v2_101
from nets.mobilenet import mobilenet_v2

slim = tf.contrib.slim


def build_resnet(model_variant,
                 weight_decay,
                 is_training,
                 fine_tune_batch_norm,
                 output_stride):
  """Factory function that returns a callable that wraps all argument settings
  and takes as input a 4-D input tensor `images`. 

  Args:
    model_variant: string scalar, model variant of ResNet.
    weight_decay: float scalar, weight decay.
    is_training: bool scalar, whether ResNet is in training model.
    fine_tune_batch_norm: bool scalar, whether to fine tune batch norm 
      parameters.
    output_stride: int scalar, output stride of the final feature map.

  Returns:
    resnet_fn: a callable that takes as input a 4-D tensor `images`. 

  Raises:
    ValueError if `model_variant` is not supported.
  """
  if model_variant == 'resnet_v1_50':
    resnet = resnet_v1_50
  elif model_variant == 'resnet_v1_101':
    resnset = resnet_v1_101
  elif model_variant == 'resnet_v2_50':
    resnet = resnet_v2_50
  elif model_variant == 'resnet_v2_101':
    resnet = resnet_v2_101
  else:
    raise ValueError('Unsupported resnet variant %s' % model_variant)

  arg_scope = resnet_arg_scope(weight_decay=weight_decay,
                                            batch_norm_decay=0.9997, 
                                            batch_norm_epsilon=1e-5,
                                            batch_norm_scale=True)

  def resnet_fn(images):
    with slim.arg_scope(arg_scope):
      return resnet(inputs=images,
                    num_classes=None,
                    is_training=(is_training and fine_tune_batch_norm),
                    global_pool=False,
                    output_stride=output_stride)

  return resnet_fn


def build_mobilenetv2(depth_multiplier,
                      weight_decay,
                      is_training,
                      fine_tune_batch_norm,
                      output_stride):
  """Factory function that returns a callable that wraps all argument settings
  and takes as input a 4-D input tensor `images`.

  Args:
    depth_multiplier: int scalar, depth multiplier for depthwise convolution in
      a separable convolution op.
    weight_decay: float scalar, weight decay.
    is_training: bool scalar, whether ResNet is in training model.
    fine_tune_batch_norm: bool scalar, whether to fine tune batch norm 
      parameters.
    output_stride: int scalar, output stride of the final feature map.

  Returns:
    mobilenetv2_fn: a callable that takes as input a 4-D tensor `images`.
  """
  def mobilenetv2_fn(images):
    with slim.arg_scope(mobilenet_v2.training_scope(weight_decay=weight_decay, 
        is_training=(is_training and fine_tune_batch_norm))):
      return mobilenet_v2.mobilenet_base(
          images,
          conv_defs=mobilenet_v2.V2_DEF,
          depth_multiplier=depth_multiplier,
          min_depth=8 if depth_multiplier == 1.0 else 1,
          divisible_by=8 if depth_multiplier == 1.0 else 1,
          final_endpoint='layer_18',
          output_stride=output_stride)
  return mobilenetv2_fn
      

def get_spatial_dims(inputs):
  """Get the static (or dynamic if static dims are not available) height and 
  width dimension of input feature map.

  Args:
    inputs: float tensor of shape [batch_size, height, width, channels].

  Returns:
    height: int scalar or int scalar tensor, height.
    width: int scalar or int scalar tensor, width.
  """
  static_height, static_width = inputs.shape.as_list()[1:3]
  if static_height is not None and static_width is not None:
    return static_height, static_width

  height, width = tf.unstack(tf.shape(inputs)[1:3])
  return height, width


def split_separable_conv2d(inputs,
                           filters,
                           kernel_size=3,
                           rate=1,
                           weight_decay=4e-5,
                           depthwise_weights_initializer_stddev=0.33,
                           pointwise_weights_initializer_stddev=0.06,
                           scope=None):
  """Separable 2D convolution split into depthwise and pointwise stages.

  Args:
    inputs: 4-D float tensor of shape [batch_size, height, width, channels].
    filters: int scalar, the depth of the output tensor.
    kernel_size: int scalar or a 2-tuple of ints, kernel size.
    rate: int scalar, atrous rate for the depthwise convolution.
    weight_decay: float scalar, weight decay.
    depthwise_weights_initializer_stddev: float scalar, stddev of depthwise
      weights.
    pointwise_weights_initializer_stddev: float scalar, stddev of pointwise
      weights.
    scope: string scalar or None, scope of the separable conv2d operation.

  Returns:
    4-D float tensor of shape [batch_size, height, width, channels]
      holding output of separable 2d convolution.
  """
  # depthwise
  outputs = slim.separable_conv2d(
      inputs,
      num_outputs=None,
      kernel_size=kernel_size,
      depth_multiplier=1,
      rate=rate,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=depthwise_weights_initializer_stddev),
      weights_regularizer=None,
      scope=scope + '_depthwise')
  # pointwise
  return slim.conv2d(
      outputs,
      filters,
      1,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=pointwise_weights_initializer_stddev),
      weights_regularizer=slim.l2_regularizer(weight_decay),
      scope=scope + '_pointwise')


def get_decoder_end_point_name(model_variant):
  """Returns the name of the end point tensor from backbone.

  Args:
    model_variant: name of the backbone model.

  Returns:
    a string scalar representing the name of the end point.

  Raises:
    ValueError if `model_variant` is not supported.
  """
  if 'resnet_v1' in model_variant:
    end_point = 'block1/unit_2/bottleneck_v1/conv3'
    return '/'.join([model_variant, end_point])
  elif 'resnet_v2' in model_variant:
    end_point = 'block1/unit_2/bottleneck_v2/conv3'
    return '/'.join([model_variant, end_point])
  elif 'mobilenet_v2' == model_variant:
    return  'layer_4/depthwise_output'
  raise ValueError('Unsupported model variant %s' % model_variant)


def build_optimizer(learning_rate, momentum=0.9):
  """Builds the optimizer and the learning rate.

  Args:
    learning_rate: int scalar tensor, learning rate.
    momentum: float scalar, momentum of MomentumOptimizer.

  Returns:
    optimizer: an Optimizer instance.
  """
  optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
  return optimizer


def build_poly_decay_lr(init_learning_rate, num_steps, learning_power=0.9):
  """Build polynomially decayed learning rate.

  Args:
    init_learning_rate: float scalar, initial learning rate.
    num_steps: int scalar, total num of steps (mini-batches)
    learning_power: float scalar, power of the polynomial decay.

  Returns:
    learning_rate: int scalar tensor, learning rate.
  """
  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.train.polynomial_decay(
      init_learning_rate,
      global_step,
      num_steps,
      end_learning_rate=0,
      power=learning_power)
  return learning_rate


def build_manual_step_lr(init_learning_rate, boundaries, rates, warmup=False):
  """Generates manually stepped learning rate schedule.

  Example:
  Given init_learning_rate = 0.1, boundaries = [10, 20], and rates [.01, .001], 
  the learning rate is returned as a scalar tensor, which equals
    .1 for global steps in interval [0, 10);
    .01 for global steps in interval [10, 20);
    .001 for global steps in interval [20, inf).
  If `warmup` is True, then the learning rate is linearly interpolated between
  .1 and .01 for global_steps in interval [0, 10).

  Note `boundaries` must be an increasing list of ints starting from a positive 
  integer, and has length `len(rates) - 1`. 

  Args:
    init_learning_rate: float scalar, initial learning rate.
    boundaries: a list of increasing ints starting from a positive int, the
      steps at which learning rate is changed.
    rates: a list of floats of length `len(boundaries) + 1`, the learning rates
      in the intervals defined by the integers in `boundaries`.
    warmup: bool scalar, whether to linearly interpolate learning rates from 
      `rates[0]` to `rates[1]` for global steps within the interval 
      `[0, boundaries[0])`.

  Returns
    learning_rate: float scalar tensor, the learning rate at global step 
      `global_step`.
  """
  rates = [init_learning_rate] + rates
  if len(rates) != len(boundaries) + 1:
    raise ValueError('`len(rates)` must be equal to `len(boundaries) + 1`.')

  if warmup:
    slope = float(rates[1] - rates[0]) / boundaries[0]
    warmup_steps = list(range(boundaries[0]))
    warmup_rates = [rates[0] + slope * step for step in warmup_steps]
    boundaries = warmup_steps[1:] + boundaries
    rates = warmup_rates + rates[1:]

  global_step = tf.train.get_or_create_global_step()
  lower_cond = tf.concat([tf.less(global_step, boundaries), [True]], 0)
  upper_cond = tf.concat([[True], tf.greater_equal(global_step, boundaries)], 0)
  indicator = tf.to_float(tf.logical_and(lower_cond, upper_cond))
  learning_rate = tf.reduce_sum(rates * indicator, name='learning_rate')
  return learning_rate


def get_vars_available_in_ckpt(variable_list, ckpt_path):
  """Returns the variables to restore from the checkpoint.

  Args:
    variable_list: a list of tf.Variables, holding the variables to restore.
    ckpt_path: string scalar, the path to the checkpoint to restore variables 
      from. 

  Returns:
    vars_int_ckpt: dict mapping from names to variables.
 
  Raises:
    ValueError if variables in memory and variables in checkpoints do not match
      on shape.  
  """
  vars_name_to_tensor = {var.op.name: var for var in variable_list}
  ckpt_reader = tf.train.NewCheckpointReader(ckpt_path)
  ckpt_vars_to_shape = ckpt_reader.get_variable_to_shape_map()
  vars_in_ckpt ={}

  for var_name, var in sorted(vars_name_to_tensor.items()):
    if var_name in ckpt_vars_to_shape:
      if ckpt_vars_to_shape[var_name] == var.shape.as_list():
        vars_in_ckpt[var_name] = var
      else:
        raise ValueError('Variable %s has shape %s, but checkpoint has shape %s'
            % (var_name, var.shape.as_list(), ckpt_vars_to_shape[var_name])) 
    else:
      logging.warning('Variable [%s] is not available in checkpoint', var_name) 
  return vars_in_ckpt


def add_loss(labels, logits, ignore_label):
  """Addes ops to compute the prediction loss (softmax cross entropy).

  Args:
    labels: int tensor of shape [batch_size, crop_height, crop_width]
    logits: float tensor of shape 
        [batch_size, crop_height, crop_width, num_classes]
    ignore_label: int scalar, the class label to be ignored in `labels`. 
  """
  masks = tf.to_float(tf.not_equal(labels, ignore_label))
  labels = tf.to_int32(tf.to_float(labels) * masks)
  tf.losses.sparse_softmax_cross_entropy(labels, logits, weights=masks)


def compute_mIOU(labels, predictions, ignore_label, num_classes):
  """Computes mean IOU.

  Args:
    labels: int tensor of shape [batch_size, height, width], holding the 
      groundtruth class labels. Populated with values from 0 (background) to 
      `num_classes`. May additionally include `ignore_label`.
    predictions: int tensor of shape [batch_size, height, width], holding the
      predicted classes (from 0 to `num_classes`) for each spatial location.
    ignore_label: int scalar, the class label to be ignored in `labels`. 
    num_classes: int scalar, num of classes.

  Returns:
    mean_iou: float scalar tensor holding computed mIOU.
    update_op: tf.Operation to update the confusion matrix 
  """
  masks = tf.to_float(tf.not_equal(labels, ignore_label))
  labels = tf.to_int32(tf.to_float(labels) * masks)

  labels = tf.reshape(labels, [-1])
  predictions = tf.reshape(predictions, [-1])
  masks = tf.reshape(masks, [-1])

  mean_iou, update_op = tf.metrics.mean_iou(
      labels, predictions, num_classes, weights=masks)
  return mean_iou, update_op

def resnet_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  """Returns arg scope for building resnet models. Has the same behavior as
  slim.nets.resnet_utils.resnet_arg_scope, except that slim.separable_conv2d
  is added to the list of ops.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }

  with slim.arg_scope(
      [slim.conv2d, slim.separable_conv2d],
      weights_regularizer=slim.regularizers.l2_regularizer(weight_decay),
      weights_initializer=slim.initializers.variance_scaling_initializer(),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc


def create_restore_saver(load_ckpt_path=None):
  """Creates saver for restoring from a checkpoint.

  Args:
    load_ckpt_path: string scalar or None, path to the ckpt from which the 
      variables are restored. If None, restore **ALL** variables. Otherwise
      restore only those found in the checkpoint.

  Returns:
    restore_saver: the restore saver.
  """
  if load_ckpt_path is None:  # Evaluator or Inferencer or loading from 
                              # a segmentation model checkpoint
    restore_saver = tf.train.Saver(tf.global_variables())
  else: # Trainer
    vars_available = get_vars_available_in_ckpt(
        slim.get_model_variables(), load_ckpt_path)
    restore_saver = tf.train.Saver(vars_available)
  return restore_saver


def create_persist_saver(max_to_keep=None):
  """Creates saver for persisting variables to a checkpoint.

  Args:
    max_to_keep: int scalar of None, max num of recent checkpoints to keep. If 
      None, keep all.

  Returns:
    persist_saver: the persist saver.
  """
  persist_saver = tf.train.Saver(max_to_keep=max_to_keep)
  return persist_saver


def visualize_predictions(predictions):
  """Convert predictions holding class labels into displayable 3-channel
  images.

  Args:
    predictions: 3-D numpy array of shape [batch_size, height, width] holding
      predicted class labels for each spatial location.

  Returns:
    predictions_image: 4-D numpy array of shape [batch_size, height, width, 3]
      where each class label prediction is mapped to RGB values.
  """
  color_map = get_color_map()
  predictions_image = np.array([color_map[i] for i in predictions.ravel()])
  predictions_image = predictions_image.reshape(predictions.shape + (3,))
  return predictions_image


def get_color_map(num_colors=256, normalized=False):
  """Creates color map.

  Args:
    num_colors: int scalar, total num of colors.
    normalized: bool scalar, whether RGB channel values are in the range of 
      [0, 1] float (True) or [0, 255] uint8 (False).
    
  Returns:
    color_map: numpy array of shape [num_colors, 3].
  """
  def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)

  dtype = 'float32' if normalized else 'uint8'
  color_map = np.zeros((num_colors, 3), dtype=dtype)

  for i in range(num_colors):
    r = g = b = 0
    c = i
    for j in range(8):
      r = r | (bitget(c, 0) << 7 - j)
      g = g | (bitget(c, 1) << 7 - j)
      b = b | (bitget(c, 2) << 7 - j)
      c = c >> 3

    color_map[i] = np.array([r, g, b])

  color_map = color_map/255 if normalized else color_map
  return color_map

