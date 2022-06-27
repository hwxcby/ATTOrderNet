from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import contextlib
import functools
import math
import random

# Dependency imports

import numpy as np


import tensorflow as tf

from tensorflow.python.eager import context as tfe_context
from tensorflow.python.framework import function
from tensorflow.python.framework import ops

allow_defun = False


def dropout_with_broadcast_dims(x, keep_prob, broadcast_dims=None, **kwargs):
  """Like tf.nn.dropout but takes broadcast_dims instead of noise_shape.

  Instead of specifying noise_shape, this function takes broadcast_dims -
  a list of dimension numbers in which noise_shape should be 1.  The random
  keep/drop tensor has dimensionality 1 along these dimensions.

  Args:
    x: a floating point tensor.
    keep_prob: A scalar Tensor with the same type as x.
      The probability that each element is kept.
    broadcast_dims: an optional list of integers
      the dimensions along which to broadcast the keep/drop flags.
    **kwargs: keyword arguments to tf.nn.dropout other than "noise_shape".
  Returns:
    A Tensor with the same size and shape as x.
  """
  assert "noise_shape" not in kwargs
  if broadcast_dims:
    shape = tf.shape(x)
    ndims = len(x.get_shape())
    kwargs["noise_shape"] = [
        1 if i in broadcast_dims else shape[i] for i in xrange(ndims)]
  return tf.nn.dropout(x, keep_prob, **kwargs)


def comma_separated_string_to_integer_list(s):
  return [int(i) for i in s.split(",") if i]


def layer_norm_compute_python(x, epsilon, scale, bias):
  """Layer norm raw computation."""
  mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
  variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
  norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
  return norm_x * scale + bias

@function.Defun(compiled=True)
def layer_norm_compute_grad(x, epsilon, scale, bias, dy):
  y = layer_norm_compute_python(x, epsilon, scale, bias)
  dx = tf.gradients(ys=[y], xs=[x, epsilon, scale, bias], grad_ys=[dy])
  return dx


@function.Defun(
    compiled=True,
    separate_compiled_gradients=True,
    grad_func=layer_norm_compute_grad)
def layer_norm_compute(x, epsilon, scale, bias):
  return layer_norm_compute_python(x, epsilon, scale, bias)


def layer_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
  """Layer normalize the tensor x, averaging over the last dimension."""
  if filters is None:
    filters = x.get_shape()[-1]
  with tf.variable_scope(
      name, default_name="layer_norm", values=[x], reuse=reuse):
    scale = tf.get_variable(
        "layer_norm_scale", [filters], initializer=tf.ones_initializer())
    bias = tf.get_variable(
        "layer_norm_bias", [filters], initializer=tf.zeros_initializer())
    if allow_defun:
      result = layer_norm_compute(x, tf.constant(epsilon), scale, bias)
      result.set_shape(x.get_shape())
    else:
      result = layer_norm_compute_python(x, epsilon, scale, bias)
    return result


def noam_norm(x, epsilon=1.0, name=None):
  """One version of layer normalization."""
  with tf.name_scope(name, default_name="noam_norm", values=[x]):
    shape = x.get_shape()
    ndims = len(shape)
    return (tf.nn.l2_normalize(x, ndims - 1, epsilon=epsilon) * tf.sqrt(
        tf.to_float(shape[-1])))


def apply_norm(x, norm_type, depth):
  """Apply Normalization."""
  if norm_type == "layer":
    return layer_norm(x, filters=depth, epsilon=1e-6)
  if norm_type == "noam":
    return noam_norm(x, 1.0)
  if norm_type == "none":
    return x
  raise ValueError("Parameter normalizer_fn must be one of: 'layer', 'batch',"
                   "'noam', 'none'.")

def layer_prepostprocess(previous_value,
                         x,
                         sequence,
                         dropout_rate,
                         norm_type,
                         depth,
                         default_name,
                         name=None,
                         dropout_broadcast_dims=None):
  """Apply a sequence of functions to the input or output of a layer.

  The sequence is specified as a string which may contain the following
  characters:
    a: add previous_value
    n: apply normalization
    d: apply dropout

  For example, if sequence=="dna", then the output is
    previous_value + normalize(dropout(x))

  Args:
    previous_value: A Tensor, to be added as a residual connection ('a')
    x: A Tensor to be transformed.
    sequence: a string.
    dropout_rate: a float
    norm_type: a string (see apply_norm())
    depth: an integer (size of last dimension of x).
    epsilon: a float (parameter for normalization)
    default_name: a string
    name: a string
    dropout_broadcast_dims:  an optional list of integers less than 3
      specifying in which dimensions to broadcast the dropout decisions.
      saves memory.

  Returns:
    a Tensor
  """
  with tf.variable_scope(name, default_name=default_name):
    if sequence == "none":
      return x
    for c in sequence:
      if c == "a":
        x += previous_value
      elif c == "n":
        x = apply_norm(x, norm_type, depth)
      else:
        assert c == "d", ("Unknown sequence step %s" % c)
        x = dropout_with_broadcast_dims(x, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
    return x


def layer_preprocess(layer_input, layer_prepostprocess_dropout):
  """Apply layer preprocessing.

  See layer_prepostprocess() for details.

  A hyperparameters object is passed for convenience.  The hyperparameters
  that may be used are:

    layer_preprocess_sequence
    layer_prepostprocess_dropout
    norm_type
    hidden_size
    norm_epsilon

  Args:
    layer_input: a Tensor
    hparams: a hyperparameters object.

  Returns:
    a Tensor
  """
  assert "a" not in "n", ("No residual connections allowed in hparams.layer_preprocess_sequence")
  return layer_prepostprocess(
      None,
      layer_input,
      sequence="n",
      dropout_rate=layer_prepostprocess_dropout,
      norm_type="layer",
      depth=None,
      dropout_broadcast_dims=comma_separated_string_to_integer_list(""),
      default_name="layer_prepostprocess")


def layer_postprocess(layer_input, layer_output, layer_prepostprocess_dropout):
  """Apply layer postprocessing.

  See layer_prepostprocess() for details.

  A hyperparameters object is passed for convenience.  The hyperparameters
  that may be used are:

    layer_postprocess_sequence
    layer_prepostprocess_dropout
    norm_type
    hidden_size
    norm_epsilon

  Args:
    layer_input: a Tensor
    layer_output: a Tensor
    hparams: a hyperparameters object.

  Returns:
    a Tensor
  """
  return layer_prepostprocess(
      layer_input,
      layer_output,
      sequence="da",
      dropout_rate=layer_prepostprocess_dropout,
      norm_type="layer",
      depth=None,
      dropout_broadcast_dims=comma_separated_string_to_integer_list(""),
      default_name="layer_postprocess")


def dense(x, units, **kwargs):
  """Identical to tf.layers.dense, Memory optimization on tpu."""
  fn = lambda x: tf.layers.dense(x, units, **kwargs)
  return fn(x)


def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i in xrange(len(static)):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret

def dense_relu_dense(inputs, filter_size, output_size, dropout=0.0,
                     dropout_broadcast_dims=None):
  """Hidden layer with RELU activation followed by linear projection."""
  h = dense(
      inputs, filter_size, use_bias=True, activation=tf.nn.relu, name="conv1")
  if dropout != 0.0:
    h = dropout_with_broadcast_dims(
        h, 1.0 - dropout, broadcast_dims=dropout_broadcast_dims)
  o = dense(h, output_size, use_bias=True, name="conv2")
  return o















