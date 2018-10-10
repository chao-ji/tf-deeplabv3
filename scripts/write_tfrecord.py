r"""Executable for converting raw image and label files into TFRecord files.

Example:
  python write_tfrecord.py \
    --list_path=/LIST/PATH/FILE.txt \
    --labels_dir=/PATH/TO/LABELS/DIR \
    --images_dir=/PATH/TO/IMAGES/DIR \
    --split=val
"""
import os
from PIL import Image
import numpy as np
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string('list_path', None, 'Path to text file containing basenames '
    '(e.g. 2007_000001) of image-label pairs.')
flags.DEFINE_string('labels_dir', None, 'Path to directory containing labels.')
flags.DEFINE_string('images_dir', None, 'Path to directory containing images.')
flags.DEFINE_string('labels_ext', '.png', 
    'Extension name of files containing labels.')
flags.DEFINE_string('images_ext', '.jpg',
    'Extension name of files containing images.')
flags.DEFINE_string('split', None, 'Split name (e.g. train, val, test).')

FLAGS = flags.FLAGS


def _bytes_list_feature(values):
  return tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[values]))


def image_label_to_tfexample(image_data, label_data):
  return tf.train.Example(features=tf.train.Features(feature={
      'image': _bytes_list_feature(image_data),
      'label': _bytes_list_feature(label_data)}))


def write_to_tfrecord_file(image_file_list, label_file_list, dataset_name, num_per_shard=1024):
  file_index = 0
  num_examples = len(image_file_list)
  num_shards = int(np.ceil(num_examples / num_per_shard))

  for shard_id in range(num_shards):
    start_index = shard_id * num_per_shard
    end_index = min((shard_id + 1) * num_per_shard, num_examples)
    print('start_index', start_index, 'end_index', end_index)

    tfrecord_writer = tf.python_io.TFRecordWriter(os.path.join(
        dataset_name, '%s-%05d-of-%05d.tfrecord' % (dataset_name, shard_id + 1, num_shards)))

    for index in range(start_index, end_index):
      image = image_file_list[index]
      label = label_file_list[index]

      image_data = tf.gfile.FastGFile(image, 'rb').read()
      label_data = tf.gfile.FastGFile(label, 'rb').read()
      example = image_label_to_tfexample(image_data, label_data)

      tfrecord_writer.write(example.SerializeToString())

    tfrecord_writer.close()


def main(_):
  basenames = [l.strip() for l in open(FLAGS.list_path)]
  image_filenames = [os.path.join(FLAGS.images_dir, l + FLAGS.images_ext) for l in basenames]
  label_filenames = [os.path.join(FLAGS.labels_dir, l + FLAGS.labels_ext) for l in basenames]

  print(image_filenames[0])
  print(label_filenames[0])
  if not tf.gfile.Exists(FLAGS.split):
    tf.gfile.MakeDirs(FLAGS.split)

  write_to_tfrecord_file(image_filenames, label_filenames, FLAGS.split)


if __name__ == '__main__':
  tf.flags.mark_flag_as_required('list_path')
  tf.flags.mark_flag_as_required('labels_dir')
  tf.flags.mark_flag_as_required('images_dir')
  tf.flags.mark_flag_as_required('split')
  tf.app.run()
