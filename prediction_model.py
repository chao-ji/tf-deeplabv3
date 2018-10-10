import tensorflow as tf

from dataset import RGB_MEAN
import utils

slim = tf.contrib.slim


class DeepLabV3PredictionModel(object):
  """DeepLabV3 prediction model.

  Implements `predict` method that takes as input 4-D tensor `images` and 
  returns `logits` tensor.
  """
  def __init__(self, 
               feature_extractor,
               feature_extractor_name,
               num_classes,
               is_training,
               fine_tune_batch_norm,
               preprocess_images_option='subtract_imagenet_mean',
               weight_decay=4e-5,
               use_image_level_feature=True,
               atrous_rates=(6, 12, 18),
               aspp_with_separable_conv=True,
               use_decoder=True,
               decoder_with_separable_conv=True):
    """Constructor.

    Args:
      feature_extractor: the backbone model (callable) that takes as input 
        `images` tensor and returns a tuple: `features` and `end_points`.
      feature_extractor_name: string scalar, name of `feature_extractor`.
      num_classes: int scalar, num of classes (including background class).
      is_training: bool scalar, whether model is in training mode (for 
        batch-norm and dropout).
      fine_tune_batch_norm: bool scalar, whether to fine tune batch norm 
        parameters.
      preprocess_images_option: string scalar or None, the method by which
        images are preprocessed (normalized). If None, images are fed to model 
        as is. 
      weight_decay: float scalar, weight decay.
      use_image_level_feature: bool scalar, whether to add the image level
        feature branch in ASPP module.
      atrous_rates: 3-tuple of ints or None, the atrous rates for ASPP module.
        If None, no atrous convolution branches are added.
      aspp_with_separable_conv: bool scalar, whether to use separable 
        convolution (True) or regular convolution in ASPP module.
      use_decoder: bool scalar, whether to use decoder module.
      decoder_with_separable_conv: bool scalar, whether to use separable
        convolution (True) or regular convolution in decoder module. Ignored
        if `use_decoder` is False.
    """
    self._feature_extractor = feature_extractor
    self._feature_extractor_name = feature_extractor_name
    self._num_classes = num_classes
    self._is_training = is_training
    self._fine_tune_batch_norm = fine_tune_batch_norm
    self._preprocess_images_option = preprocess_images_option
    self._weight_decay = weight_decay
    self._use_image_level_feature = use_image_level_feature
    self._atrous_rates = atrous_rates
    self._aspp_with_separable_conv = aspp_with_separable_conv
    self._use_decoder = use_decoder
    self._decoder_with_separable_conv = decoder_with_separable_conv

  def predict(self, images):
    """Runs the input tensor `images` through the forward pass to get the output
    `logits` tensor.

    Args:
      images: 4-D float tensor of shape [batch_size, height, width, channels]

    Returns:
      logits: 4-D float tensor of shape [batch_size, height, width, num_classes]
    """
    # Normalize images 
    images = self._preprocess_images(images)
    # Backbone
    features, end_points = self._feature_extractor(images)
    # ASPP
    aspp_output = self._build_atrous_spatial_pyramid_pooling(features)
    # Decoder (Optional)
    decoder_output = self._build_decoder(aspp_output, end_points)
    # Logits
    logits_spatial_dims = utils.get_spatial_dims(images)
    logits = self._build_logits(decoder_output, logits_spatial_dims)
    return logits

  def _preprocess_images(self, images):
    """Normalize images by either subtracting the image net mean or mapping the
    pixel values in range [-1, 1].
    
    Args:
      images: 4-D float tensor of shape [batch_size, height, width, channels]

    Returns:
      preprocessed_images: 4-D float tensor of shape 
        [batch_size, height, width, channels]
    """
    if self._preprocess_images_option == 'subtract_imagenet_mean':
      return images - tf.reshape(RGB_MEAN, [1, 1, 1, 3])
    elif self._preprocess_images_option == 'zero_mean_unit_range':
      return images * 2 / 255.0 - 1.0
    elif self._preprocess_images_option is None:
      return images
    raise ValueError('Unsupported preprocess option %s' % 
        self._preprocess_images_option)

  def _build_atrous_spatial_pyramid_pooling(self, features, depth=256):
    """Builds the ASPP module containing up to five branches:
    1. 1x1 convolution
    3. 3x3 atrous convolution with different rates.
    3. Globally pooled image level features.

    Args:
      features: 4-D float tensor of shape [batch_size, height, width, channels]
      depth: int scaler, all branches are projected into `depth`-dimension, and 
        the final output has depth `depth`.

    Returns:
      aspp_output: 4-D float tensor of shape [batch_size, height, width, depth]
    """
    with slim.arg_scope(utils.resnet_arg_scope(
        weight_decay=self._weight_decay, batch_norm_decay=0.9997)):
      with slim.arg_scope([slim.batch_norm], 
          is_training=self._is_training and self._fine_tune_batch_norm):
        features_list = []

        # 1. 1x1 convolution
        features_list.append(slim.conv2d(features, depth, 1, scope='aspp0'))
        # 2. 3x3 convolution with atrous_rates[0, 1, 2]
        if self._atrous_rates:
          for i, rate in enumerate(self._atrous_rates, 1):
            scope = 'aspp%s' % i
            if self._aspp_with_separable_conv:
              aspp_features = utils.split_separable_conv2d(
                  features,
                  filters=depth,
                  rate=rate,
                  weight_decay=self._weight_decay,
                  scope=scope)
            else:
              aspp_features = slim.conv2d(
                  features, depth, 3, rate=rate, scope=scope)
            features_list.append(aspp_features)

        # 3. Image level feature
        if self._use_image_level_feature:
          image_feature = tf.reduce_mean(features, [1, 2], keepdims=True)
          image_feature = slim.conv2d(image_feature, depth, 1,
              scope='image_pooling')
          resize_height, resize_width = utils.get_spatial_dims(features)
          image_feature = tf.image.resize_images(image_feature,
              [resize_height, resize_width],
              method=tf.image.ResizeMethod.BILINEAR)
          features_list.append(image_feature)

        aspp_output = tf.concat(features_list, axis=3)
        aspp_output = slim.conv2d(
            aspp_output, depth, 1, stride=1, scope='concat_projection')

        aspp_output = slim.dropout(
            aspp_output,
            keep_prob=0.9,
            is_training=self._is_training,
            scope='concat_projection_dropout')
    return aspp_output

  def _build_decoder(self, features, end_points, decoder_depth=256):
    """Optionally builds decoder module.

    Args:
      features: 4-D float tensor of shape [batch_size, height, width, channels],
        output of ASPP module.
      end_points: a dict mapping for end_point names to end_point tensors 
        returned by the feature extractor.
      decoder_depth: int scalar, the output has depth dimension `decoder_depth`.

    Returns:
      decoder_features: 4-D float tensor of shape 
        [batch_size, height, width, decoder_depth]
    """
    if not self._use_decoder:
      return features

    with slim.arg_scope(utils.resnet_arg_scope(
        weight_decay=self._weight_decay, batch_norm_decay=0.9997)):
      with slim.arg_scope([slim.batch_norm],
          is_training=self._is_training and self._fine_tune_batch_norm):
        with tf.variable_scope('decoder', values=[features]): 
    
          end_point_name = utils.get_decoder_end_point_name(
              self._feature_extractor_name)
    
          early_layer = end_points[end_point_name]
          
          early_layer = slim.conv2d(early_layer, 48, 1, scope='feature_projection0')  
          resize_height, resize_width = utils.get_spatial_dims(early_layer)
     
          resized_features = tf.image.resize_images(features,
              [resize_height, resize_width],
              method=tf.image.ResizeMethod.BILINEAR) 
          concat_features = tf.concat([resized_features, early_layer], 3)
          
          if self._decoder_with_separable_conv:
            decoder_features = utils.split_separable_conv2d(
                concat_features,
                filters=decoder_depth,
                weight_decay=self._weight_decay,
                scope='decoder_conv0')
            decoder_features = utils.split_separable_conv2d(
                decoder_features,
                filters=decoder_depth,
                weight_decay=self._weight_decay,
                scope='decoder_conv1')
          else: 
            decoder_features = slim.conv2d(concat_features, decoder_depth, 3)
            decoder_features = slim.conv2d(decoder_features, decoder_depth, 3) 
    return decoder_features
 
  def _build_logits(self, features, logits_spatial_dims):
    """Builds the logits module.

    Args:
      features: 4-D float tensor of shape [batch_size, height, width, channels],
        output of decoder module.
      logits_spatial_dims: a list of 2 ints or int scalars [height, width], 
        holding the spatial dimensions of a feature map. 

    Returns:
      upsampled_logits: 4-D float tensor of shape 
        [batch_size, height, width, num_classes]
    """
    with slim.arg_scope(utils.resnet_arg_scope(
        weight_decay=self._weight_decay)):
      with tf.variable_scope('logits', values=[features]): 
        logits = slim.conv2d(features, self._num_classes, 1, 
            activation_fn=None, normalizer_fn=None, scope='semantic')
        upsampled_logits = tf.image.resize_images(logits, logits_spatial_dims,
            method=tf.image.ResizeMethod.BILINEAR)
    return upsampled_logits

