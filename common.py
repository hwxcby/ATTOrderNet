from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensor2tensor.utils import expert_utils
import common_attention
import common_layers

from tensorflow.python.util import nest
from collections import defaultdict
import contextlib
import functools
import math
import random

# Dependency imports

import numpy as np

from tensorflow.python.eager import context as tfe_context
from tensorflow.python.framework import function
from tensorflow.python.framework import ops


def transformer_prepare_encoder(inputs):
  """Prepare one shard of the model for the encoder.

  Args:
    inputs: a Tensor.  #(b_sz, tstps_en+1, hidden)

  Returns:
    encoder_input: a Tensor, bottom of encoder stack
    encoder_self_attention_bias: a bias tensor for use in encoder self-attention

  """
  ishape_static = inputs.shape.as_list()
  encoder_input = inputs

  encoder_padding = common_attention.embedding_to_padding(encoder_input)
  ignore_padding = common_attention.attention_bias_ignore_padding(encoder_padding)
  encoder_self_attention_bias = ignore_padding

  return (encoder_input, encoder_self_attention_bias)


def transformer_encoder(encoder_input,
                        encoder_self_attention_bias,
                        name="encoder",
                        nonpadding=None,
                        save_weights_to=None,
                        make_image_summary=False,
                        num_hidden_layers=6,
                        hidden_size=512,
                        num_heads=8,
                        attention_dropout=0.1,
                        layer_prepostprocess_dropout=0.1,
                        filter_size=2048,
                        relu_dropout=0.1,
                        length=10,
                        batch=16):
  """A stack of transformer layers.

  Args:
    encoder_input: a Tensor
    encoder_self_attention_bias: bias Tensor for self-attention
       (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This must either be
      passed in, which we do for "packed" datasets, or inferred from
      encoder_self_attention_bias.  The knowledge about padding is used
      for pad_remover(efficiency) and to mask out padding in convoltutional
      layers.
    save_weights_to: an optional dictionary to capture attention weights
      for vizualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.

  Returns:
    y: a Tensors
  """
  x = encoder_input     # (b_sz, tstps_en+1, hidden)
  b_sz=tf.shape(encoder_input)[0]
  attention_dropout_broadcast_dims = (common_layers.comma_separated_string_to_integer_list(""))
  with tf.variable_scope(name):
    if nonpadding is not None:
      padding = 1.0 - nonpadding
    else:
      padding = common_attention.attention_bias_to_padding(encoder_self_attention_bias)
      nonpadding = 1.0 - padding
    pad_remover = None
    pad_remover = expert_utils.PadRemover(padding)
    for layer in xrange(num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        with tf.variable_scope("self_attention"):
          y = common_attention.multihead_attention(
              common_layers.layer_preprocess(x, layer_prepostprocess_dropout),
              None,
              encoder_self_attention_bias,
              hidden_size,
              hidden_size,
              hidden_size,
              num_heads,
              attention_dropout,
              attention_type="dot_product",
              save_weights_to=save_weights_to,
              max_relative_position=0,
              make_image_summary=make_image_summary,
              dropout_broadcast_dims=attention_dropout_broadcast_dims)
          x = common_layers.layer_postprocess(x, y, layer_prepostprocess_dropout)
        with tf.variable_scope("ffn"):
          y = transformer_ffn_layer(
              common_layers.layer_preprocess(x, layer_prepostprocess_dropout), pad_remover,
              nonpadding_mask=nonpadding, filter_size=filter_size, hidden_size=hidden_size, relu_dropout=relu_dropout)
          x = common_layers.layer_postprocess(x, y, layer_prepostprocess_dropout)

    encoder_output = common_layers.layer_preprocess(x, layer_prepostprocess_dropout)   # (b_sz, tstps_en+1, hidden)

    encoder_output_final = encoder_output

    return encoder_output_final


def transformer_ffn_layer(x,
                          pad_remover=None,
                          nonpadding_mask=None,
                          filter_size=2048,
                          hidden_size=512,
                          relu_dropout=0.1):
  """Feed-forward layer in the transformer.

  Args:
    x: a Tensor of shape [batch_size, length, hparams.hidden_size]
    hparams: hyperparmeters for model
    pad_remover: an expert_utils.PadRemover object tracking the padding
      positions. If provided, when using convolutional settings, the padding
      is removed before applying the convolution, and restored afterward. This
      can give a significant speedup.
    conv_padding: a string - either "LEFT" or "SAME".
    nonpadding_mask: an optional Tensor with shape [batch_size, length].
      needed for convolutoinal layers with "SAME" padding.
      Contains 1.0 in positions corresponding to nonpadding.

  Returns:
    a Tensor of shape [batch_size, length, hparams.hidden_size]
  """
  ffn_layer = "dense_relu_dense"
  relu_dropout_broadcast_dims = (common_layers.comma_separated_string_to_integer_list(""))
  if ffn_layer == "dense_relu_dense":
    # In simple convolution mode, use `pad_remover` to speed up processing.
    if pad_remover:
      original_shape = common_layers.shape_list(x)
      # Collapse `x` across examples, and remove padding positions.
      x = tf.reshape(x, tf.concat([[-1], original_shape[2:]], axis=0))
      x = tf.expand_dims(pad_remover.remove(x), axis=0)
    conv_output = common_layers.dense_relu_dense(
        x,
        filter_size,
        hidden_size,
        dropout=relu_dropout,
        dropout_broadcast_dims=relu_dropout_broadcast_dims)
    if pad_remover:
      # Restore `conv_output` to the original shape of `x`, including padding.
      conv_output = tf.reshape(
          pad_remover.restore(tf.squeeze(conv_output, axis=0)), original_shape)
    return conv_output




def w_encoder_attention(queries,   
                        keys,
                        sequence_length,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        scope="w_encoder_attention",
                        reuse=None):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      if num_units is None:
          num_units = queries.get_shape().as_list[-1]

      Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
      K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C) # (b_sz*tstps_en, len_sen, hidden_size)
      V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
      x = K * Q
      x = tf.reshape(x, [tf.shape(x)[0],tf.shape(x)[1],num_heads, int(num_units/num_heads)])
      outputs = tf.transpose(tf.reduce_sum(x, 3),[0,2,1])
      outputs = outputs / (K.get_shape().as_list()[-1] ** 0.5)
      key_masks = tf.sequence_mask(sequence_length, tf.shape(keys)[1], dtype=tf.float32)
      key_masks = tf.reshape(tf.tile(key_masks,[1, num_heads]),[tf.shape(key_masks)[0],num_heads,tf.shape(key_masks)[1]])

      paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
      outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
      outputs = tf.nn.softmax(outputs, 2)
      V_ = tf.reshape(V, [tf.shape(V)[0],tf.shape(V)[1], num_heads, int(num_units/num_heads)])
      V_ = tf.transpose(V_, [0,2,1,3])

      weight = outputs
      outputs = tf.reshape(tf.reduce_sum(V_ * tf.expand_dims(outputs, -1),2),[-1,num_units])  # (b_sz*tstps_en, hidden_size)
      outputs = tf.nn.dropout(outputs, 1-dropout_rate, name='w_encoder_attention_dropout')
  return outputs














