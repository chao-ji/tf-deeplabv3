r"""Executable for evaluating DeepLabV3 model for semantic segmentation.
Example:

  python run_evaluator.py \
    --model_variant=resnet_v2_101 \
    --filenames=/PATH/TO/VAL_TFRECORD_FILE1 \
    --filenames=/PATH/TO/VAL_TFRECORD_FILE2 \
    --ckpt_path=/PATH/TO/DEEPLABV3_MODEL.ckpt
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import glob

import utils
from dataset import EvaluatorDeepLabV3Dataset
from prediction_model import DeepLabV3PredictionModel
from model_runners import DeepLabV3Evaluator

flags = tf.app.flags


flags.DEFINE_integer('ignore_label', 255, 'Index of class in labels to be '
    'ignored when computing loss')

flags.DEFINE_enum('model_variant', 'resnet_v2_101', ['resnet_v2_50',
    'resnet_v2_101', 'resnet_v1_50', 'resnet_v1_101', 'mobilenet_v2'],
    'Name of the backbone feature extractor.')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight decay.')
flags.DEFINE_enum('output_stride', '16', ['8', '16', '32'], 'Output stride.')

flags.DEFINE_integer('num_classes', 21,
    'Num of classes (including background class)')
flags.DEFINE_enum('preprocess_images_option', 'subtract_imagenet_mean',
    ['subtract_imagenet_mean', 'zero_mean_unit_range'],
    'Option by which images are preprocessed.')
flags.DEFINE_bool('use_image_level_feature', True,
    'Whether to use image level features.')
flags.DEFINE_multi_integer('atrous_rates', None,
    'Rates (list of three integers) of atrous convolution in ASPP module.')
flags.DEFINE_bool('aspp_with_separable_conv', False,
    'Whether to use separable convolution in ASPP module.')
flags.DEFINE_bool('use_decoder', False, 'Whether to use decoder module.')
flags.DEFINE_bool('decoder_with_separable_conv', True,
    'Whether to use separable convolution in decoder module.')

flags.DEFINE_multi_string('filenames', None,
    'List of input TFRecord file names.')
flags.DEFINE_string('ckpt_path', None,
    'Path to checkpoint file to which model variables are saved.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
  dataset = EvaluatorDeepLabV3Dataset()

  if 'resnet' in FLAGS.model_variant:
    feature_extractor_fn = utils.build_resnet(
        model_variant=FLAGS.model_variant,
        weight_decay=FLAGS.weight_decay,
        is_training=False,
        fine_tune_batch_norm=False,
        output_stride=int(FLAGS.output_stride))
  elif 'mobilenet' in FLAGS.model_variant:
    feature_extractor_fn = utils.build_mobilenetv2(
        depth_multiplier=1,
        weight_decay=FLAGS.weight_decay,
        is_training=False,
        fine_tune_batch_norm=False,
        output_stride=int(FLAGS.output_stride))
  else:
    raise ValueError('Unsupported feature extractor %s' % FLAGS.model_variant)

  model = DeepLabV3PredictionModel(feature_extractor_fn, 
      FLAGS.model_variant, 
      FLAGS.num_classes,
      is_training=False,
      fine_tune_batch_norm=False,
      preprocess_images_option=FLAGS.preprocess_images_option,
      weight_decay=FLAGS.weight_decay,
      use_image_level_feature=FLAGS.use_image_level_feature,
      atrous_rates=FLAGS.atrous_rates,
      aspp_with_separable_conv=FLAGS.aspp_with_separable_conv,
      use_decoder=FLAGS.use_decoder,
      decoder_with_separable_conv=FLAGS.decoder_with_separable_conv)

  evaluator = DeepLabV3Evaluator(model, 
                                 num_classes=FLAGS.num_classes, 
                                 ignore_label=FLAGS.ignore_label)

  to_be_run_dict = evaluator.evaluate(filenames=FLAGS.filenames,
                                      dataset=dataset)
  miou = to_be_run_dict.pop('mean_iou')

  restore_saver = utils.create_restore_saver()

  with tf.Session() as sess:
    restore_saver.restore(sess, FLAGS.ckpt_path)
    sess.run(tf.local_variables_initializer())

    losses = []
    while True:
      try:
        result_dict = sess.run(to_be_run_dict)
        losses.append(result_dict['total_loss'])

      except tf.errors.OutOfRangeError:
        break

    print('Num of images evaluated:', len(losses))
    print('Loss:', np.mean(losses))
    print('mIOU:', sess.run(miou))


if __name__ == '__main__':
  tf.flags.mark_flag_as_required('filenames')
  tf.flags.mark_flag_as_required('ckpt_path')
  tf.app.run()

