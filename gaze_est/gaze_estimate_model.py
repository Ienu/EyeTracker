"""DeepLab v3 models based on slim library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers

#from utils import preprocessing
from gaze_est.utils import preprocessing

_BATCH_NORM_DECAY = 0.9997
_WEIGHT_DECAY = 5e-4


def relu6(x, name='relu6'):
    return tf.nn.relu6(x, name)


def hard_swish(x, name='hard_swish'):
    with tf.variable_scope(name):
        h_swish = x * tf.nn.relu6(x + 3) / 6
    return h_swish


def hard_sigmoid(x, name='hard_sigmoid'):
    with tf.variable_scope(name):
        h_sigmoid = tf.nn.relu6(x + 3) / 6
    return h_sigmoid

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _batch_normalization_layer(inputs, momentum=0.997, epsilon=1e-3, is_training=True, name='bn', reuse=None):
    return tf.layers.batch_normalization(inputs=inputs,
                                         momentum=momentum,
                                         epsilon=epsilon,
                                         scale=True,
                                         center=True,
                                         training=is_training,
                                         name=name,
                                         reuse=reuse)


def _conv2d_layer(inputs, filters_num, kernel_size, name, use_bias=False, strides=1, reuse=None, padding="SAME"):
    conv = tf.layers.conv2d(
        inputs=inputs, filters=filters_num,
        kernel_size=kernel_size, strides=[strides, strides], kernel_initializer=tf.glorot_uniform_initializer(),
        padding=padding, #('SAME' if strides == 1 else 'VALID'),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4), use_bias=use_bias, name=name,
        reuse=reuse)
    return conv


def _conv_1x1_bn(inputs, filters_num, name, use_bias=True, is_training=True, reuse=None):
    kernel_size = 1
    strides = 1
    x = _conv2d_layer(inputs, filters_num, kernel_size, name=name + "/conv", use_bias=use_bias, strides=strides)
    x = _batch_normalization_layer(x, momentum=0.997, epsilon=1e-3, is_training=is_training, name=name + '/bn',
                                   reuse=reuse)
    return x


def _conv_bn_relu(inputs, filters_num, kernel_size, name, use_bias=True, strides=1, padding="SAME", is_training=True, activation=relu6, reuse=None):
    x = _conv2d_layer(inputs, filters_num, kernel_size, name, use_bias=use_bias, strides=strides, reuse=reuse, padding=padding)
    x = _batch_normalization_layer(x, momentum=0.997, epsilon=1e-3, is_training=is_training, name=name + '/bn', reuse=reuse)
    x = activation(x)
    return x


def _dwise_conv(inputs, k_h=3, k_w=3, depth_multiplier=1, strides=(1, 1),
                padding='SAME', name='dwise_conv', use_bias=False,
                reuse=None):
    kernel_size = (k_w, k_h)
    in_channel = inputs.get_shape().as_list()[-1]
    filters = int(in_channel*depth_multiplier)
    return tf.layers.separable_conv2d(inputs, filters, kernel_size,
                                      strides=strides, padding=padding,
                                      data_format='channels_last', dilation_rate=(1, 1),
                                      depth_multiplier=depth_multiplier, activation=None,
                                      use_bias=use_bias, name=name, reuse=reuse
                                      )



def _fully_connected_layer(inputs, units, name="fc", activation=None, use_bias=True, reuse=None):
    return tf.layers.dense(inputs, units, activation=activation, use_bias=use_bias,
                           name=name, reuse=reuse)


def _global_avg(inputs, pool_size, strides, padding='valid', name='global_avg'):
    return tf.layers.average_pooling2d(inputs, pool_size, strides,
                                       padding=padding, data_format='channels_last', name=name)


def _squeeze_excitation_layer(input, out_dim, ratio, layer_name, is_training=True, reuse=None):
    with tf.variable_scope(layer_name, reuse=reuse):
        squeeze = _global_avg(input, pool_size=input.get_shape()[1:-1], strides=1)

        excitation = _fully_connected_layer(squeeze, units=out_dim / ratio, name=layer_name + '_excitation1',
                                            reuse=reuse)
        excitation = relu6(excitation)
        excitation = _fully_connected_layer(excitation, units=out_dim, name=layer_name + '_excitation2', reuse=reuse)
        excitation = hard_sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = input * excitation
        return scale


def mobilenet_v3_block(input, k_s, expansion_ratio, output_dim, stride, name, is_training=True,
                       use_bias=True, shortcut=True, activatation="RE", ratio=16, se=False,
                       reuse=None):
    bottleneck_dim = expansion_ratio

    with tf.variable_scope(name, reuse=reuse):
        # pw mobilenetV3
        net = _conv_1x1_bn(input, bottleneck_dim, name="pw", use_bias=use_bias)

        if activatation == "HS":
            net = hard_swish(net)
        elif activatation == "RE":
            net = relu6(net)
        else:
            raise NotImplementedError

        # dw
        net = _dwise_conv(net, k_w=k_s, k_h=k_s, strides=[stride, stride], name='dw',
                          use_bias=use_bias, reuse=reuse)

        net = _batch_normalization_layer(net, momentum=0.997, epsilon=1e-3,
                                         is_training=is_training, name='dw_bn', reuse=reuse)

        if activatation == "HS":
            net = hard_swish(net)
        elif activatation == "RE":
            net = relu6(net)
        else:
            raise NotImplementedError

        # squeeze and excitation
        if se:
            channel = net.get_shape().as_list()[-1]
            net = _squeeze_excitation_layer(net, out_dim=channel, ratio=ratio, layer_name='se_block')

        # pw & linear
        net = _conv_1x1_bn(net, output_dim, name="pw_linear", use_bias=use_bias)

        # element wise add, only for stride==1
        if shortcut and stride == 1:
            net += input
            net = tf.identity(net, name='block_output')

    return net


def mobilenet_v3_small(inputs, classes_num, multiplier=1.0, is_training=True, reuse=None):
    end_points = {}
    layers = [
        # ic oc  kw s  nl    se    exp
        [16, 16, 3, 2, "RE", True, 16],
        [16, 24, 3, 2, "RE", False, 72],
        [24, 24, 3, 1, "RE", False, 88],
        [24, 40, 5, 2, "RE", True, 96],
        [40, 40, 5, 1, "RE", True, 240],
        [40, 40, 5, 1, "RE", True, 240],
        [40, 48, 5, 1, "HS", True, 120],
        [48, 48, 5, 1, "HS", True, 144],
        [48, 96, 5, 2, "HS", True, 288],
        [96, 96, 5, 1, "HS", True, 576],
        [96, 96, 5, 1, "HS", True, 576],
    ]
    #inputs = tf.convert_to_tensor(inputs)
    input_size = inputs.shape[2]
    assert ((input_size % 32 == 0) and (input_size % 32 == 0))

    reduction_ratio = 4
    with tf.variable_scope('init', reuse=reuse):
        # init_conv_out = _make_divisible(16 * multiplier)
        x = _conv_bn_relu(inputs, filters_num=16, kernel_size=3, name='init',
                          use_bias=True, strides=2, is_training=is_training, activation=hard_swish)

    with tf.variable_scope("MobilenetV3_small", reuse=reuse):
        for idx, (in_channels, out_channels, kernel_size, stride, activatation, se, exp_size) in enumerate(layers):
            # in_channels = _make_divisible(in_channels * multiplier)
            # out_channels = _make_divisible(out_channels * multiplier)
            # exp_size = _make_divisible(exp_size * multiplier)
            x = mobilenet_v3_block(x, kernel_size, exp_size, out_channels, stride,
                                   "bneck{}".format(idx), is_training=is_training, use_bias=True,
                                   shortcut=(in_channels==out_channels), activatation=activatation,
                                   ratio=reduction_ratio, se=se)
            end_points["bneck{}".format(idx)] = x

        # conv1_in = _make_divisible(96 * multiplier)
        # conv1_out = _make_divisible(576 * multiplier)
    #     x = _conv_bn_relu(x, filters_num=576, kernel_size=1, name="conv1_out",
    #                       use_bias=True, strides=1, is_training=is_training, activation=hard_swish)
    #
    #     x = _squeeze_excitation_layer(x, out_dim=576, ratio=reduction_ratio, layer_name="conv1_out",
    #                                  is_training=is_training, reuse=None)
    #     end_points["conv1_out_1x1"] = x
    #
    #     x = _global_avg(x, pool_size=x.get_shape()[1:-1], strides=1)
    #     # x = hard_swish(x)
    #     end_points["global_pool"] = x
    #
    # with tf.variable_scope('Logits_out', reuse=reuse):
    #     # conv2_in = _make_divisible(576 * multiplier)
    #     # conv2_out = _make_divisible(1280 * multiplier)
    #     x = _conv2d_layer(x, filters_num=1280, kernel_size=1, name="conv2", use_bias=True, strides=1)
    #     x = hard_swish(x)
    #     end_points["conv2_out_1x1"] = x
    #
    #     x = _conv2d_layer(x, filters_num=classes_num, kernel_size=1, name="conv3", use_bias=True, strides=1)
    #     logits = tf.layers.flatten(x)
    #     logits = tf.identity(logits, name='output')
    #     end_points["Logits_out"] = logits

    return x, end_points


def gaze_estimate_generator(batch_norm_decay,
                              data_format='channels_last'):
  """Generator for DeepLab v3 plus models.

  Args:
    num_classes: The number of possible classes for image classification.
    output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
      the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
    base_architecture: The architecture of base Resnet building block.
    pre_trained_model: The path to the directory that contains pre-trained models.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
      Only 'channels_last' is supported currently.

  Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the DeepLab v3 model.
  """
  if data_format is None:
    # data_format = (
    #     'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
    pass

  if batch_norm_decay is None:
    batch_norm_decay = _BATCH_NORM_DECAY

  def model(inputs, is_training):
    """Constructs the ResNet model given the inputs."""
    if data_format == 'channels_first':
      raise ValueError("must be channels_last")

    # tf.logging.info('net shape: {}'.format(inputs.shape))
    # encoder
    end_points = {}
    with tf.variable_scope("eyes_processing"):
        left_eye_logits, left_eye_end_points = mobilenet_v3_small(inputs[0], 2, is_training=is_training, reuse=False)
        # print('left_eye_end_points  ', left_eye_end_points)
        left_net = left_eye_end_points['bneck10']
        right_eye_logits, right_eye_end_points = mobilenet_v3_small(inputs[1], 2, is_training=is_training, reuse=True)
        right_net = right_eye_end_points['bneck10']
        eye_net = tf.concat([left_net, right_net], axis=3)
        eye_net = _conv_bn_relu(eye_net, filters_num=96, kernel_size=1, name="pw1",is_training=is_training, use_bias=True)
        eye_net = _squeeze_excitation_layer(eye_net, out_dim=96, ratio=4, layer_name="eye_conv1_out",
                                      is_training=is_training, reuse=None)
        eye_net = _conv_bn_relu(eye_net, filters_num=48, kernel_size=1, name="pw2",is_training=is_training, use_bias=True)
        # eye_net = _conv_bn_relu(eye_net, filters_num=384, kernel_size=2, name='conv2', is_training=is_training,
        #                           use_bias=True, strides=2, padding='valid')
        end_points["eye_out"] = eye_net

    with tf.variable_scope('face_processing'):
        face_logits, face_end_points = mobilenet_v3_small(inputs[2], 2, is_training=is_training, reuse=False)
        face_net = face_end_points['bneck10']
        #face_mask = tf.convert_to_tensor(inputs[3])
        face_mask = _conv_bn_relu(inputs[3], filters_num=8, kernel_size=3, name='mask_conv1',
                          use_bias=True, strides=2, is_training=is_training, activation=hard_swish)
        face_mask = _conv_bn_relu(face_mask, filters_num=16, kernel_size=3, name='mask_conv2',
                                  use_bias=True, strides=2, is_training=is_training, activation=hard_swish)
        face_mask = _conv_bn_relu(face_mask, filters_num=24, kernel_size=3, name='mask_conv3',
                                  use_bias=True, strides=3, padding='valid', is_training=is_training, activation=hard_swish)

        facem_net = tf.concat([face_net, face_mask], axis=3)
        facem_net = _conv_bn_relu(facem_net, filters_num=96, kernel_size=1, name="pw1", is_training=is_training, use_bias=True)
        facem_net = _squeeze_excitation_layer(facem_net, out_dim=96, ratio=4, layer_name="facem_conv1_out",
                                      is_training=is_training, reuse=None)
        facem_net = _conv_bn_relu(facem_net, filters_num=48, kernel_size=1, name="pw2", is_training=is_training, use_bias=True)
        # facem_net = _conv_bn_relu(facem_net, filters_num=384, kernel_size=2, name='conv2', is_training=is_training,
        #                          use_bias=True, strides=2, padding='valid')
        end_points["facem_out"] = facem_net

    with tf.variable_scope('gaze_estimate'):
        gaze_net = tf.concat([eye_net, facem_net], axis=3)
        gaze_net = _conv_bn_relu(gaze_net, filters_num=48, kernel_size=2, name="pw1", is_training=is_training,
                                 use_bias=True,strides=2, padding='valid')
        gaze_net = _squeeze_excitation_layer(gaze_net, out_dim=48, ratio=4, layer_name="conv1_out", reuse=None)
        gaze_net = _conv_bn_relu(gaze_net, filters_num=24, kernel_size=1, name='conv2', is_training=is_training,
                                  use_bias=True, strides=1, padding='valid')
        logits = _conv2d_layer(gaze_net, filters_num=2, kernel_size=1, name='conv3',
                               use_bias=True, strides=1, padding='valid')
        end_points['gaze'] = logits

    print('gaze net ', end_points)
    return logits

  return model


def gaze_estimate_model_fn(features, labels, mode, params):
  """Model function for PASCAL VOC."""
  if isinstance(features, dict):
    features = features['feature']

  # images = tf.cast(
  #     tf.map_fn(preprocessing.mean_image_addition, features),
  #     tf.uint8)

  network = gaze_estimate_generator(params['batch_norm_decay'])

  logits = network(features, mode == tf.estimator.ModeKeys.TRAIN)

  pred_coor = tf.reshape(logits, [-1, 2])
  # pred_coor = tf.squeeze(logits)

  predictions = {
      'coordinate': pred_coor,
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Delete 'decoded_labels' from predictions because custom functions produce error when used with saved_model

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'preds': tf.estimator.export.PredictOutput(
                predictions)
        })

  labels = tf.squeeze(labels)  # reduce the channel dimension.
  predictions['preds'] = pred_coor
  predictions['labels'] = labels
  mse_loss = tf.losses.mean_squared_error(labels, pred_coor)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(mse_loss, name='mse_loss')
  tf.summary.scalar('mse_loss', mse_loss)

  if not params['freeze_batch_norm']:
    train_var_list = [v for v in tf.trainable_variables()]
  else:
    train_var_list = [v for v in tf.trainable_variables()
                      if 'beta' not in v.name and 'gamma' not in v.name]

  # Add weight decay to the loss.
  with tf.variable_scope("total_loss"):
    loss = mse_loss + params.get('weight_decay', _WEIGHT_DECAY) * tf.add_n(
        [tf.nn.l2_loss(v) for v in train_var_list])
  # loss = tf.losses.get_total_loss()  # obtain the regularization losses as well

  if mode == tf.estimator.ModeKeys.TRAIN:
    # tf.summary.image('images',
    #                  tf.concat(axis=2, values=[images, gt_decoded_labels, pred_decoded_labels]),
    #                  max_outputs=params['tensorboard_images_max_outputs'])  # Concatenate row-wise.

    global_step = tf.train.get_or_create_global_step()

    if params['learning_rate_policy'] == 'piecewise':
      # Scale the learning rate linearly with the batch size. When the batch size
      # is 128, the learning rate should be 0.1.
      initial_learning_rate = 0.1 * params['batch_size'] / 128
      batches_per_epoch = params['num_train'] / params['batch_size']
      # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
      boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
      values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
      learning_rate = tf.train.piecewise_constant(
          tf.cast(global_step, tf.int32), boundaries, values)
    elif params['learning_rate_policy'] == 'poly':
      learning_rate = tf.train.polynomial_decay(
          params['initial_learning_rate'],
          tf.cast(global_step, tf.int32) - params['initial_global_step'],
          params['max_iter'], params['end_learning_rate'], power=params['power'])
    else:
      raise ValueError('Learning rate policy must be "piecewise" or "poly"')

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    # optimizer = tf.train.MomentumOptimizer(
    #     learning_rate=learning_rate,
    #     momentum=params['momentum'])

    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate)

    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step, var_list=train_var_list)
      px_error = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(labels, pred_coor), axis=1)))
      # px_error = tf.metrics.accuracy(
      #    labels, pred_coor)
      # Create a tensor named train_accuracy for logging purposes
      tf.identity(px_error, name='train_px_error')
      # metrics = {'px_error': px_error}
      tf.summary.scalar('train_px_error', px_error)

      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions,
          loss=loss,
          train_op=train_op,
          eval_metric_ops=None)

  if mode == tf.estimator.ModeKeys.EVAL:
      px_error = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(labels, pred_coor), axis=1)))
      # Create a tensor named train_accuracy for logging purposes
      tf.identity(px_error, name='val_px_error')
      # metrics = {'px_error': px_error}
      tf.summary.scalar('val_px_error', px_error)

      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions,
          loss=loss,
          eval_metric_ops=None)
