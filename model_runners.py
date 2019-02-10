from abc import abstractproperty
from abc import ABCMeta

import tensorflow as tf

import utils

slim = tf.contrib.slim


class BaseModelRunner(object):
  """Base model runner to be subclassed by Trainer, Evaluator, Inferencer."""
  __metaclass__ = ABCMeta

  @abstractproperty
  def mode(self):
    """Returns a string scalar indicating mode of model (train, eval or infer).
    """
    pass

  def check_dataset_mode(self, dataset):
    """Checks if mode (train, eval, or infer) of dataset and model match.

    Args:
      dataset: a DeepLabV3Dataset instance.

    Raises:
      ValueError if mode of `dataset` and `self` do not match.
    """
    if dataset.mode != self.mode:
      raise ValueError('mode of dataset({}) and model({}) do not match.'
          .format(dataset.mode, self.mode))


class DeepLabV3Trainer(BaseModelRunner):
  """DeepLabV3 model trainer."""
  def __init__(self, prediction_model, ignore_label=255):
    """Constructor.

    Args:
      prediction_model: a DeepLabV3PredictionModel instance.
      ignore_label: int scalar, integer representing the class in `labels` to 
        be ignored (i.e. masked out when computing loss). 
    """
    self._prediction_model = prediction_model
    self._ignore_label = ignore_label

  @property
  def mode(self):
    return tf.contrib.learn.ModeKeys.TRAIN

  def train(self, filenames, dataset, optimizer, learning_rate):
    """Adds training related ops to the graph.

    Args:
      filenames: list of strings, the list of TFRecord filenames.
      dataset: a TrainerDeepLabV3Dataset instance.
      optimizer: an Optimizer instance.
      learning_rate: float tensor scalar, learning rate.

    Returns:
      to_be_run_dict: a dict mapping from names to tensors/ops 
        { 'grouped_update_op': gradient update ops and batch_norm update ops,
          'total_loss': sum of prediction loss and regularization loss,
            float scalar tensor,
          'global_step': global step, int scalar tensor,
          'summary_op': string scalar tensor, serialized summary message.}
    """
    self.check_dataset_mode(dataset)

    with tf.device('/device:CPU:0'):
      tensor_dict = dataset.get_tensor_dict(filenames)
  
    with slim.arg_scope(
        [slim.model_variable, slim.variable], device='/device:CPU:0'): 
      with tf.device('/device:GPU:0'):
        logits = self._prediction_model.predict(tensor_dict['images'])

    with tf.device('/device:GPU:0'):
      utils.add_loss(tensor_dict['labels'], logits, self._ignore_label)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      pred_loss = tf.get_collection(tf.GraphKeys.LOSSES)
      reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

      total_loss = tf.add_n(pred_loss + reg_losses)

      global_step = tf.train.get_or_create_global_step()

    with tf.device('/device:GPU:0'):
      grads_and_vars = optimizer.compute_gradients(total_loss)

    with tf.device('/device:CPU:0'):
      grad_update_op = optimizer.apply_gradients(
          grads_and_vars, global_step=global_step)

      update_ops.append(grad_update_op)

      grouped_update_op = tf.group(*update_ops, name='update_barrier')
    with tf.control_dependencies([grouped_update_op]):
#      with tf.device('/device:GPU:0'):
      total_loss = tf.identity(total_loss, name='total_loss')

      summary_op = tf.summary.merge([
          tf.summary.scalar('total_loss', total_loss),
          tf.summary.scalar('learning_rate', learning_rate)])

    to_be_run_dict = {'grouped_update_op': grouped_update_op, 
                      'total_loss': total_loss, 
                      'global_step': global_step,                      
                      'summary_op': summary_op}
    return to_be_run_dict


class DeepLabV3Evaluator(BaseModelRunner):
  """DeepLabV3 model evaluator."""
  def __init__(self, prediction_model, num_classes, ignore_label=255):
    """Constructor.

    Args:
      prediction_model: a DeepLabV3PredictionModel instance.
      num_classes: int scalar, num of classes (including background class).
      ignore_label: int scalar, integer representing the class in `labels` to 
        be ignored (i.e. masked out when computing loss).
    """
    self._prediction_model = prediction_model
    self._num_classes = num_classes
    self._ignore_label = ignore_label

  @property
  def mode(self):
    return tf.contrib.learn.ModeKeys.EVAL

  def evaluate(self, filenames, dataset):
    """Adds evaluation related ops to the graph.

    Args:
      filenames: list of strings, the list of TFRecord filenames.
      dataset: a EvaluatorDeepLabV3Dataset instance.

    Returns:
      to_be_run_dict: a dict mapping from names to tensors/ops 
        { 'total_loss': sum of prediction loss and regularization loss,
            float scalar tensor,
          'mean_iou': mean IOU, float scalar tensor,
          'miou_update_op': tf.Operation to update the confusion matrix 
        }
    """
    self.check_dataset_mode(dataset)

    tensor_dict = dataset.get_tensor_dict(filenames)
    logits = self._prediction_model.predict(tensor_dict['images'])

    utils.add_loss(tensor_dict['labels'], logits, self._ignore_label)
    pred_loss = tf.get_collection(tf.GraphKeys.LOSSES)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(pred_loss + reg_losses)

    predictions = tf.argmax(logits, axis=3, output_type=tf.int64)

    mean_iou, update_op = utils.compute_mIOU(tensor_dict['labels'], predictions,
        self._ignore_label, self._num_classes)

    to_be_run_dict = {'total_loss': total_loss, 
                      'mean_iou': mean_iou, 
                      'miou_update_op': update_op}
    return to_be_run_dict


class DeepLabV3Inferencer(BaseModelRunner):
  """DeepLabV3 model inferencer."""
  def __init__(self, prediction_model):
    """Constructor.

    Args:
      prediction_model: a DeepLabV3PredictionModel instance.
    """
    self._prediction_model = prediction_model

  @property
  def mode(self):
    return tf.contrib.learn.ModeKeys.INFER

  def infer(self, filenames, dataset):
    """Adds inference related ops to the graph.

    Args:
      filenames: list of strings, the list of TFRecord filenames.
      dataset: a InferencerDeepLabV3Dataset instance.

    Returns:
      to_be_run_dict: a dict mapping from names to tensors
        {'predictions': predicted labels, int tensor of shape 
            [batch_size, height, width],
         'filename': filename of input image, string scalar}
    """
    self.check_dataset_mode(dataset)

    tensor_dict = dataset.get_tensor_dict(filenames)
    logits = self._prediction_model.predict(tensor_dict['images'])
    predictions = tf.cast(
        tf.argmax(logits, axis=3, output_type=tf.int32), tf.uint8)
    to_be_run_dict = {'predictions': predictions, 
                      'filename': tensor_dict['filename']}
    return to_be_run_dict
