""""""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.layers.python.layers import layers

ATROUS_RATES = 6, 12, 18


class DeepLabV3Model(object):
  """DeepLabV3 Model builder."""
  def __init__(self, hparams, dataset, mode):
    self._mode = mode
    self._is_training = self._mode == tf.contrib.learn.ModeKeys.TRAIN
    self._output_stride = hparams.output_stride
    self._image_size = hparams.crop_image_size
    self._strided_image_size = tuple((i - 1) // hparams.output_stride + 1
        for i in self._image_size)

    if hparams.backbone_model == "resnet-50":
      self._backbone_model = resnet_v2.resnet_v2_50
    else:
      self._backbone_model = resnet_v2.resnet_v2_101
 
    if hparams.output_stride == 16:
      self._atrous_rates = ATROUS_RATES
    else:
      self._atrous_rates = [2 * rate for rate in ATROUS_RATES]

    self._logits = self._build_graph(hparams, dataset)


  def _build_graph(self, hparams, dataset):
    backbone_model = self._backbone_model

    num_classes = hparams.num_classes
    weight_decay = hparams.weight_decay
    batch_norm_decay = hparams.batch_norm_decay

    images = dataset.images

    with arg_scope(resnet_v2.resnet_arg_scope(
        weight_decay=weight_decay,
        batch_norm_decay=batch_norm_decay)):

      # `net` is the BN-relu transformed output of `block4` of ResNet
      # when `num_classes` is None and `global_pool` is False.
      net, _ = backbone_model(
          images,
          num_classes=None,
          is_training=self._is_training,
          global_pool=False,
          output_stride=self._output_stride)

      aspp = self._atrous_spatial_pyramid_pooling(net)

      logits = _final_logits(aspp, self._image_size, num_classes)

    return logits

  def _atrous_spatial_pyramid_pooling(self, inputs, depth=256, scope=None):
    rates = self._atrous_rates

    with tf.variable_scope(scope, "ASPP", values=[inputs]):
      with arg_scope([layers.batch_norm], is_training=self._is_training):
        conv_1x1 = layers_lib.conv2d(
            inputs, depth, 1, stride=1, scope="conv_1x1")
        conv_3x3_1 = layers_lib.conv2d(
            inputs, depth, 3, stride=1, rate=rates[0], scope="conv_3x3_1")
        conv_3x3_2 = layers_lib.conv2d(
            inputs, depth, 3, stride=1, rate=rates[1], scope="conv_3x3_2")
        conv_3x3_3 = layers_lib.conv2d(
            inputs, depth, 3, stride=1, rate=rates[2], scope="conv_3x3_3")

        with tf.variable_scope("image_level_features"):
          image_level_features = tf.reduce_mean(
              inputs, [1, 2], keepdims=True, name="global_pooling")
          image_level_features = layers_lib.conv2d(
              image_level_features, depth, [1, 1], stride=1)
          image_level_features = tf.image.resize_bilinear(
              image_level_features, self._strided_image_size, name="upsampling")

        stacked = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3,
            image_level_features], axis=3)

        outputs = layers_lib.conv2d(
            stacked, depth, 1, stride=1, scope="projection")

    return outputs


def _get_atrous_rates(output_stride):
  if output_stride == 16:
    return ATROUS_RATES
  else:
    return [2 * rate for rate in ATROUS_RATES]


def _final_logits(inputs, outputs_size, num_classes, scope=None):
  with tf.variable_scope(scope, "logits", values=[inputs]):
    logits = layers_lib.conv2d(inputs,
        num_classes, 1, stride=1, activation_fn=None, normalizer_fn=None)
    logits = tf.image.resize_bilinear(logits, outputs_size)
  return logits
    
