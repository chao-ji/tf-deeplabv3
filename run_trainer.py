r"""Executable for training DeepLabV3 model for semantic segmentation.
Example:
  python run_trainer.py \
    --model_variant=resnet_v2_101 \
    --filenames=/PATH/TO/TFRECORD_FILE1 \
    --filenames=/PATH/TO/TFRECORD_FILE2 \
    --filenames=/PATH/TO/TFRECORD_FILE3 \
    --load_ckpt_path=/PATH/TO/CLASSIFICATION_MODEL.ckpt \
    --output_path=/OUTPUT/PATH
"""
import sys
import os

import tensorflow as tf
import numpy as np

import utils
from dataset import TrainerDeepLabV3Dataset
from prediction_model import DeepLabV3PredictionModel
from model_runners import DeepLabV3Trainer

flags = tf.app.flags


flags.DEFINE_integer('batch_size', 12, 'Batch size.')
flags.DEFINE_bool('restore_from_seg_model', False, 'Whether to restore from a '
    'segmentation model (T) or classification model (F). Default is False.')
flags.DEFINE_bool('pad_imagenet_mean', False, 'Whether to pad the training ' 
    'images with image net mean (True) or [127.5, 127.5, 127.5] (False).')
flags.DEFINE_float('min_rescale_factor', 0.5, 'Lower bound of the factor by '
    ' which training images are rescaled.')
flags.DEFINE_float('max_rescale_factor', 2.0, 'Upper bound of the factor by '
    ' which training images are rescaled.')
flags.DEFINE_integer('crop_size_height', 513, 'Height of crop size of image.')
flags.DEFINE_integer('crop_size_width', 513, 'Width of crop_size of image.')
flags.DEFINE_integer('ignore_label', 255, 'Index of class in labels to be '
    'ignored when computing loss')

flags.DEFINE_enum('model_variant', 'resnet_v2_101', ['resnet_v2_50', 
    'resnet_v2_101', 'resnet_v1_50', 'resnet_v1_101', 'mobilenet_v2'], 
    'Name of the backbone feature extractor.')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight decay.')
flags.DEFINE_bool('fine_tune_batch_norm', True, 
    'Whether to fine tune batch norm parameters.')
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

flags.DEFINE_float('init_learning_rate', 7e-3, 'Initial learning rate.')
flags.DEFINE_integer('num_steps', 30000, 'Num of training steps.')
flags.DEFINE_float('learning_power', 0.9, 
    'Power of learning rate\'s polynomial decay.')
flags.DEFINE_float('momentum', 0.9, 'Momentum of MomentumOptimizer.')

flags.DEFINE_multi_string('filenames', None, 
    'List of input TFRecord file names.')
flags.DEFINE_string('load_ckpt_path', None, 'Path to the checkpoint holding the'
    ' weights of a trained backbone classification model (e.g. ResNet).')
flags.DEFINE_integer('log_per_steps', 200, 
    'Every `log_per_steps` to output statistics.')
flags.DEFINE_integer('persist_per_steps', 1000,
    'Every `persist_per_steps` to persist model variables.')
flags.DEFINE_string('output_path', '/tmp/deeplabv3', 
    'Path to checkpoint files and summaries are saved.')
flags.DEFINE_multi_integer('boundaries', [60000, 80000], 'A list of increasing'
    ' ints starting from a positive int, the steps at which learning rate is '
    'changed.')
flags.DEFINE_multi_float('rates', [7e-4, 7e-5], 'A list of floats of '
    'length `len(FLAGS.boundaries)`, the new learning rates at boundaries '
    'defined in `FLAGS.boundaries`.')
flags.DEFINE_bool('man_step_lr', False, 'Whether to use manual stepping (T) or '
    'polynomial decay learning rate (F). Defaults to False.')

FLAGS = flags.FLAGS


def main(_):
  dataset = TrainerDeepLabV3Dataset(batch_size=FLAGS.batch_size,
                                    pad_imagenet_mean=FLAGS.pad_imagenet_mean,
                                    min_rescale_factor=FLAGS.min_rescale_factor,
                                    max_rescale_factor=FLAGS.max_rescale_factor,
                                    crop_height=FLAGS.crop_size_height, 
                                    crop_width=FLAGS.crop_size_width, 
                                    ignore_label=FLAGS.ignore_label)
  if 'resnet' in FLAGS.model_variant:
    feature_extractor_fn = utils.build_resnet(
        model_variant=FLAGS.model_variant,
        weight_decay=FLAGS.weight_decay,
        is_training=True,
        fine_tune_batch_norm=FLAGS.fine_tune_batch_norm,
        output_stride=int(FLAGS.output_stride))
  elif 'mobilenet' in FLAGS.model_variant:
    feature_extractor_fn = utils.build_mobilenetv2(
        depth_multiplier=1,
        weight_decay=FLAGS.weight_decay,
        is_training=True,
        fine_tune_batch_norm=FLAGS.fine_tune_batch_norm,
        output_stride=int(FLAGS.output_stride))
  else:
    raise ValueError('Unsupported feature extractor %s' % FLAGS.model_variant)

  model = DeepLabV3PredictionModel(feature_extractor_fn, 
      FLAGS.model_variant, 
      FLAGS.num_classes,
      is_training=True,
      fine_tune_batch_norm=FLAGS.fine_tune_batch_norm,
      preprocess_images_option=FLAGS.preprocess_images_option,
      weight_decay=FLAGS.weight_decay,
      use_image_level_feature=FLAGS.use_image_level_feature,
      atrous_rates=FLAGS.atrous_rates,
      aspp_with_separable_conv=FLAGS.aspp_with_separable_conv,
      use_decoder=FLAGS.use_decoder,
      decoder_with_separable_conv=FLAGS.decoder_with_separable_conv)

  trainer = DeepLabV3Trainer(model, ignore_label=FLAGS.ignore_label)


  if FLAGS.man_step_lr:
    learning_rate = utils.build_manual_step_lr(
        FLAGS.init_learning_rate, list(FLAGS.boundaries), list(FLAGS.rates))
  else:
    learning_rate = utils.build_poly_decay_lr(
        FLAGS.init_learning_rate, FLAGS.num_steps, FLAGS.learning_power)

  optimizer = utils.build_optimizer(learning_rate, momentum=FLAGS.momentum) 

  to_be_run_dict = trainer.train(filenames=FLAGS.filenames,
                                 dataset=dataset, 
                                 optimizer=optimizer,
                                 learning_rate=learning_rate)

  if FLAGS.restore_from_seg_model:
    restore_saver = utils.create_restore_saver()
  else:
    restore_saver = utils.create_restore_saver(FLAGS.load_ckpt_path)

  persist_saver = utils.create_persist_saver()
  initializers = tf.global_variables_initializer()

  summary_writer = tf.summary.FileWriter(FLAGS.output_path)

  with tf.Session() as sess:
    sess.run(initializers)
    restore_saver.restore(sess, FLAGS.load_ckpt_path)

    losses = []
    while True:
      result_dict = sess.run(to_be_run_dict)
      gs = result_dict['global_step']
      if gs > FLAGS.num_steps:
        break

      losses.append(result_dict['total_loss'])
      summary_writer.add_summary(result_dict['summary_op'], gs)

      if gs % FLAGS.log_per_steps == 0:
        print('step: %d, loss: %f' % (gs, np.mean(losses)))
        losses = []
        sys.stdout.flush()
      if gs % FLAGS.persist_per_steps == 0:
        persist_saver.save(
            sess, os.path.join(FLAGS.output_path, 'model.ckpt'), global_step=gs)

    persist_saver.save(sess, os.path.join(FLAGS.output_path, 'model.ckpt'))
  summary_writer.close()

if __name__ == '__main__':
  tf.flags.mark_flag_as_required('filenames')
  tf.flags.mark_flag_as_required('load_ckpt_path')
  tf.app.run()

