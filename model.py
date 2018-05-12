"""DeelLab v3 for semantic segmentation.

[1] Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam
    Rethinking Atrous Convolution for Semantic Image Segmentation. 
    arXiv:1706.05587
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.layers.python.layers import layers

TRAIN_MODE = tf.contrib.learn.ModeKeys.TRAIN
ATROUS_RATES = 6, 12, 18


class DeepLabV3Model(object):
  """DeepLabV3 Model builder."""
  def __init__(self, hparams, dataset, mode, scope=None):
    self._mode = mode
    self._is_training = self._mode == TRAIN_MODE
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
      self._atrous_rates = tuple(2 * rate for rate in ATROUS_RATES)

    self._logits, self._preds = self._build_graph(hparams, dataset)


  def _build_graph(self, hparams, dataset):
    """Builds the graph for the forward pass (from `dataset.images` to `logits`)
    """
    backbone_model = self._backbone_model

    num_classes = hparams.num_classes
    weight_decay = hparams.weight_decay
    batch_norm_decay = hparams.batch_norm_decay

    images = dataset.images
    image_size = _get_spatial_size(images)

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

      if self._mode == TRAIN_MODE:
        _init_from_ckpt(hparams.model_ckpt_path)

      aspp = self._atrous_spatial_pyramid_pooling(net)
      logits = _final_logits(aspp, image_size, num_classes)

    preds = tf.argmax(logits, axis=3, output_type=tf.int32, name="predictions")

    return logits, preds

  def _atrous_spatial_pyramid_pooling(self, inputs, depth=256, scope=None):
    """Builds the graph for atrous SPP."""
    rates = self._atrous_rates
    strided_image_size = _get_spatial_size(inputs)

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
              image_level_features, depth, 1, stride=1, scope="projection")
          image_level_features = tf.image.resize_bilinear(
              image_level_features, strided_image_size, name="upsampling")

        stacked = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3,
            image_level_features], axis=3)

        outputs = layers_lib.conv2d(
            stacked, depth, 1, stride=1, scope="projection")

    return outputs


def _final_logits(inputs, outputs_size, num_classes, scope=None):
  """Final layer that performs a 1x1 conv to project input feature map 
  to a depth of `num_classes` and bilinearly upsample to original image size.
  """
  with tf.variable_scope(scope, "logits", values=[inputs]):
    logits = layers_lib.conv2d(inputs,
        num_classes, 1, stride=1, activation_fn=None, normalizer_fn=None)
    logits = tf.image.resize_bilinear(logits, outputs_size)
  return logits


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
