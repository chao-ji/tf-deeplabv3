"""DeelLab v3 for semantic segmentation.

[1] Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam
    Rethinking Atrous Convolution for Semantic Image Segmentation. 
    arXiv:1706.05587
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import resnet_v2

slim = tf.contrib.slim
TRAIN_MODE = tf.contrib.learn.ModeKeys.TRAIN


class DeepLabV3Model(object):
  """DeepLabV3 Model builder."""
  def __init__(self, hparams, dataset, mode, scope=None):
    self._mode = mode
    self._is_training = self._mode == TRAIN_MODE
    self._output_stride = hparams.output_stride

    if hparams.backbone_model == "resnet-50":
      self._backbone_model = resnet_v2.resnet_v2_50
    else:
      self._backbone_model = resnet_v2.resnet_v2_101
 
    if hparams.output_stride == 16:
      self._atrous_rates = hparams.atrous_rates
    else:
      self._atrous_rates = tuple(2 * rate for rate in hparams.atrous_rates)

    self._logits, self._preds = self._build_graph(hparams, dataset)


  def _build_graph(self, hparams, dataset):
    """Builds the graph for the forward pass (from `dataset.images` to `logits`)
    """
    backbone_model = self._backbone_model
    is_training = self._is_training and hparams.fine_tune_batch_norm
    spatial_size = _get_spatial_size(dataset.images)

    with slim.arg_scope(resnet_v2.resnet_arg_scope(
        weight_decay=hparams.weight_decay,
        batch_norm_decay=hparams.batch_norm_decay)):

      net, end_points = backbone_model(
          dataset.images,
          num_classes=None,
          is_training=is_training,
          global_pool=False,
          output_stride=hparams.output_stride,
          multi_grid=hparams.multi_grid)

      if self._mode == TRAIN_MODE:
        _init_from_ckpt(hparams.model_ckpt_path)

      aspp = _atrous_spatial_pyramid_pooling(
          net, self._atrous_rates, is_training)
      decoded = _decode(aspp, end_points, hparams.low_level_name)

    logits = _final_logits(decoded, spatial_size, hparams.num_classes)

    preds = tf.argmax(logits, axis=3, output_type=tf.int32, name="predictions")

    return logits, preds


def _atrous_spatial_pyramid_pooling(
    inputs, rates, is_training, depth=256, scope=None):
  """Builds the graph for atrous SPP."""
  spatial_size = _get_spatial_size(inputs)

  with tf.variable_scope(scope, "ASPP", values=[inputs]):
    with slim.arg_scope([slim.batch_norm], is_training=is_training):
      conv_1x1 = slim.conv2d(
          inputs, depth, 1, stride=1, scope="conv_1x1")
      conv_3x3_1 = slim.conv2d(
          inputs, depth, 3, stride=1, rate=rates[0], scope="conv_3x3_1")
      conv_3x3_2 = slim.conv2d(
          inputs, depth, 3, stride=1, rate=rates[1], scope="conv_3x3_2")
      conv_3x3_3 = slim.conv2d(
          inputs, depth, 3, stride=1, rate=rates[2], scope="conv_3x3_3")

      with tf.variable_scope("image_level_features"):
        image_level = tf.reduce_mean(
            inputs, [1, 2], keepdims=True, name="global_pooling")
        image_level = slim.conv2d(
            image_level, depth, 1, stride=1, scope="projection")
        image_level = tf.image.resize_bilinear(
            image_level, spatial_size, align_corners=True, name="upsampling")

      stacked = tf.concat([
          conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level], axis=3)

      outputs = slim.conv2d(
          stacked, depth, 1, stride=1, scope="projection")

  return outputs


def _decode(inputs, end_points, low_level_name,
    low_level_depth=48, depth=256, scope=None):
  """"""
  with tf.variable_scope(scope, "decoder", values=[inputs]):
    low_level = end_points[low_level_name]
    low_level = slim.conv2d(low_level, low_level_depth, 1, scope="projection")

    spatial_size = _get_spatial_size(low_level)

    high_level = tf.image.resize_bilinear(
        inputs, spatial_size, align_corners=True, name="unsampling")
  
    stacked = tf.concat([low_level, high_level], axis=3)

    decoded = _split_separable_conv2d(
        stacked, depth, rate=1, scope="decoder_conv1")
    decoded = _split_separable_conv2d(
        decoded, depth, rate=1, scope="decoder_conv2")

    return decoded
 

def _final_logits(inputs, outputs_size, num_classes, scope=None):
  """Final layer that performs a 1x1 conv to project input feature map 
  to a depth of `num_classes` and bilinearly upsample to original image size.
  """
  with tf.variable_scope(scope, "logits", values=[inputs]):
    logits = slim.conv2d(inputs,
        num_classes, 1, stride=1, activation_fn=None, normalizer_fn=None)
    logits = tf.image.resize_bilinear(logits, outputs_size)
  return logits


def _split_separable_conv2d(inputs,
                            filters,
                            rate=1,
                            weight_decay=0.00004,
                            depthwise_weights_initializer_stddev=0.33,
                            pointwise_weights_initializer_stddev=0.06,
                            scope=None):
  outputs = slim.separable_conv2d(
      inputs,
      None,
      3,
      depth_multiplier=1,
      rate=rate,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=depthwise_weights_initializer_stddev),
      weights_regularizer=None,
      scope=scope + '_depthwise')
  return slim.conv2d(
      outputs,
      filters,
      1,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=pointwise_weights_initializer_stddev),
      weights_regularizer=slim.l2_regularizer(weight_decay),
      scope=scope + '_pointwise')


def _init_from_ckpt(resnet_ckpt_path):
  """Add operations that initialize variables defined within the scope of 
  backbone resnet from a checkpoint. Note that this would override their 
  existing initializers.
  """
  with tf.name_scope("init_from_ckpt"):
    vars_to_restore = tf.global_variables()
    mapping = {v.name.split(':')[0]: v for v in vars_to_restore}
    tf.train.init_from_checkpoint(resnet_ckpt_path, mapping)


def _get_spatial_size(inputs):
  with tf.name_scope("spatial_size"):
    size = tf.shape(inputs)[1:3]
  return size
