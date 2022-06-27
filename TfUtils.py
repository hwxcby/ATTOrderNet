import tensorflow as tf
from tensorflow.python.util import nest
import numpy as np

def mkMask(input_tensor, maxLen):
    shape_of_input = tf.shape(input_tensor)  # lengths  decoder_tstps (batch,)
    shape_of_output = tf.concat([shape_of_input, [maxLen]],0)
    
    oneDtensor = tf.reshape(input_tensor, shape=(-1,))
    flat_mask = tf.sequence_mask(oneDtensor, maxlen=maxLen)
    return tf.reshape(flat_mask, shape_of_output)

def reduce_avg(reduce_tensor, mask_tensor, lengths_tensor, dim=-2):
    """
    Args:
        reduce_tensor : which tensor to average dtype float point
        mask_tensor   : same shape as reduce_tensor
        lengths_tensor : same rank as tf.reduce_sum(reduce_tensor * mask_tensor, reduction_indices=k)
        dim : which dim to average
    """
    red_sum = tf.reduce_sum(reduce_tensor * tf.to_float(mask_tensor), reduction_indices=[dim], keep_dims=False)
    red_avg = red_sum / (tf.to_float(lengths_tensor) + 1e-20)
    return red_avg

def reduce_avg_sample(reduce_target, lengths, dim):
    """
    Args:
        reduce_target : shape(d_0, d_1,..,d_dim, .., d_k) logits
        lengths : shape(d0, .., d_(dim-1))
        dim : which dimension to average, should be a python number
    """
    shape_of_lengths = lengths.get_shape()
    shape_of_target = reduce_target.get_shape()
    if len(shape_of_lengths) != dim:
        raise ValueError(('Second input tensor should be rank %d, ' +
                         'while it got rank %d') % (dim, len(shape_of_lengths)))
    if len(shape_of_target) < dim+1 :
        raise ValueError(('First input tensor should be at least rank %d, ' +
                         'while it got rank %d') % (dim+1, len(shape_of_target)))

    rank_diff = len(shape_of_target) - len(shape_of_lengths) - 1
    mxlen = tf.shape(reduce_target)[dim] # tstp_en
    mask = mkMask(lengths, mxlen)
    if rank_diff!=0:
        len_shape = tf.concat(axis=0, values=[tf.shape(lengths), [1]*rank_diff])
        mask_shape = tf.concat(axis=0, values=[tf.shape(mask), [1]*rank_diff])
    else:
        len_shape = tf.shape(lengths)
        mask_shape = tf.shape(mask)
    lengths_reshape = tf.reshape(lengths, shape=len_shape)
    mask = tf.reshape(mask, shape=mask_shape)

    mask_target = reduce_target * tf.cast(mask, dtype=reduce_target.dtype)

    red_sum = tf.reduce_sum(mask_target, axis=[dim], keep_dims=False)
    red_avg = red_sum / (tf.to_float(lengths_reshape) + 1e-30)
    return red_avg

def seq_loss_sample(logits, label, lengths):
    """
    Args
        logits: shape (b_sz, tstp)
        label: shape (b_sz, tstp)
        lengths: shape(b_sz)
    Return
        loss: A scalar tensor, mean error
    """

    loss_all = logits  # shape(b_sz, tstp)
    loss_avg = reduce_avg_sample(loss_all, lengths, dim=1)    # shape(b_sz) example level
    # loss = tf.reduce_mean(loss_avg)
    return loss_avg





    
