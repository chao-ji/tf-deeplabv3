r"""Executable for making inferences using a Trained DeepLabV3 model.
Example:

  python run_inferencer.py \
    --input_file_list=/PATH/TO/TEXT_FILE.txt \
    --ckpt_path=/PATH/TO/DEEPLABV3_MODEL.ckpt
"""
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image

import utils
from dataset import InferencerDeepLabV3Dataset
from prediction_model import DeepLabV3PredictionModel
from model_runners import DeepLabV3Inferencer

flags = tf.app.flags

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

flags.DEFINE_string('input_file_list', None,
    'Path to text file holding filenames of input images.')
flags.DEFINE_string('ckpt_path', None,
    'Path to checkpoint file to which model variables are saved.')
flags.DEFINE_string('output_path', '/tmp/deeplabv3/pred/',
    'Path to directory to which prediction images will be written.')

FLAGS = flags.FLAGS


def main(_):
  dataset = InferencerDeepLabV3Dataset()

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

  inferencer = DeepLabV3Inferencer(model)

  filenames = [line.strip() for line in open(FLAGS.input_file_list)]
  to_be_run_dict = inferencer.infer(filenames=filenames, dataset=dataset)
  restore_saver = utils.create_restore_saver()
  if not tf.gfile.Exists(FLAGS.output_path):
    tf.gfile.MakeDirs(FLAGS.output_path)

  with tf.Session() as sess:
    restore_saver.restore(sess, FLAGS.ckpt_path)
    while True:
      try:
        result_dict = sess.run(to_be_run_dict)
      except tf.errors.OutOfRangeError:
        break
      predictions = utils.visualize_predictions(result_dict['predictions'])
      out_filename = os.path.join(FLAGS.output_path, 
          os.path.basename(result_dict['filename'].decode('utf-8'))) + '.png'

      Image.fromarray(np.squeeze(predictions)).save(out_filename)
      print('Prediction written to', out_filename)


if __name__ == '__main__':
  tf.flags.mark_flag_as_required('input_file_list')
  tf.flags.mark_flag_as_required('ckpt_path')
  tf.app.run()

