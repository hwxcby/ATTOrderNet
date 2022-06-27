from Config import Config
from helper import Vocab
from myRNN import raw_rnn
from myRNN import beam_search_rnn
from myRNN import raw_rnn_rl
import helper
import TfUtils
import common

import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
import numpy as np
import sys
import os
import argparse
import logging
import cPickle as pkl
import time
from blaze.expr.expressions import shape
import core_rnn_cell_impl as rnn_cell

args=None
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training options")
    
    parser.add_argument('--load-config', action='store_true', dest='load_config', default=False)
    parser.add_argument('--weight-path', action='store', dest='weight_path', required=True)
    parser.add_argument('--debug-enable', action='store_true', dest='debug_enable', default=False)

    parser.add_argument('--gpu-num', action='store', dest='gpu_num', default=0, type=int)
    parser.add_argument('--train-test', action='store', dest='train_test', default='test', choices=['train', 'test'])
    
    args = parser.parse_args()

noize_num=0
embed_path = './all_data_sind/embed/embedding.100.txt'
train_data = './all_data_sind/train.txt'
val_data = './all_data_sind/valid.txt'
test_data = './all_data_sind/test.txt'
vocab_path = './all_data_sind/vocab.100k'

def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
    '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term

class PtrNet(object):
  
    def __init__(self, args=args, test=False):
        self.vocab = Vocab()
        self.config=Config()



        self.weight_path = args.weight_path
            
        if args.load_config == False:
            self.config.saveConfig(self.weight_path+'/config')
            print 'default configuration generated, please specify --load-config and run again.'
            sys.exit()
        else:
            self.config.loadConfig(self.weight_path+'/config')
        
        self.step_p_epoch = self.load_data(test)
        print 'pre_trained', self.config.pre_trained
        print 'max_epochs', self.config.max_epochs 
        print 'batch_size', self.config.batch_size 
        print 'embed_size', self.config.embed_size  
        print 'hidden_size', self.config.hidden_size
        print 'beam_size', self.config.beam_size

        self.add_placeholders()
        self.add_embedding()
        self.fetch_input()
        logits, self.prediction, self.beam_outputs = self.add_model()
        train_loss, valid_loss, reg_loss = self.add_loss_op(logits)
        self.train_op = self.add_train_op(train_loss)
        self.loss = valid_loss
        self.reg_loss = reg_loss
        
        MyVars = [v for v in tf.trainable_variables()]
        MyVars_name = [v.name for v in MyVars]
        self.MyVars = MyVars
        print MyVars_name

    def decoder(self, cell, initial_state, encoder_outputs, encoder_inputs ,scope=None):
        batch_size = tf.shape(encoder_outputs)[0]
        tstps_en = tf.shape(encoder_outputs)[1]
        hidden_size = tf.shape(encoder_outputs)[2]
        emb_sz = tf.shape(encoder_inputs)[2]
        
        def loop_fn(time, cell_output, cell_state, loop_state, emit_ta):
            """cell_output shape (batch_size, hidden_size)
                encoder_outputs shape (b_sz, tstps_en, h_sz)
                encoder_inputs shape (b_sz, tstps_en, embed_sz)
            """
            
            if cell_output is None:  # time == 0
                next_cell_state = initial_state
                emit_output= tf.ones(tf.shape(initial_state[1])[:1], dtype=tf.int32) * tf.constant(-1) #(batch_size)
                next_input = tf.squeeze(self.sos, [1])
                elements_finished = tf.logical_and(tf.cast(emit_output, dtype=tf.bool), False)
                
            else:
                
                next_cell_state = cell_state
                decoder_outputs = tf.expand_dims(cell_output, 1) #(batch_size, 1, hidden_size)
                encoder_outputs_reshape = tf.reshape(encoder_outputs, shape=(-1, self.config.hidden_size)) #(batch_size*time_steps, hidden_size)
                decoder_outputs_reshape = tf.reshape(decoder_outputs, shape=(-1, self.config.hidden_size)) #(batch_size*1, hidden_size)
                encoder_outputs_linear_reshape = linear(encoder_outputs_reshape, output_size=self.config.hidden_size, 
                                         scope='Ptr_W1')    #(b_sz*tstps_en, h_sz)
                decoder_outputs_linear_reshape = linear(decoder_outputs_reshape, output_size=self.config.hidden_size, 
                                         scope='Ptr_W2')    #(b_sz*1, h_sz)
                encoder_outputs_linear = tf.reshape(encoder_outputs_linear_reshape, tf.shape(encoder_outputs))
                decoder_outputs_linear = tf.reshape(decoder_outputs_linear_reshape, tf.shape(decoder_outputs))
                
                encoder_outputs_linear_expand = tf.expand_dims(encoder_outputs_linear, 1) #(b_sz, 1, tstp_en, h_sz)
                decoder_outputs_linear_expand = tf.expand_dims(decoder_outputs_linear, 2) #(b_sz, 1, 1, h_sz)
                
                after_add = tf.tanh(encoder_outputs_linear_expand + decoder_outputs_linear_expand)  #(b_sz, 1, tstp_en, h_sz)
                
                after_add_reshape = tf.reshape(after_add, shape=(-1, self.config.hidden_size))
                
                after_add_linear_reshape = linear(after_add_reshape, output_size=1, #(b_sz*1*tstp_en, 1)
                                         scope='Ptr_v')
                after_add_linear = tf.reshape(after_add_linear_reshape, shape=(-1, tstps_en)) #(b_sz, tstp_en)
                en_length_mask = tf.sequence_mask(self.encoder_tstps,                #(b_sz, tstp_en)
                                    maxlen=tstps_en, dtype=tf.bool)

                """mask out already hitted ids""" 
                hit_ids = tf.cond(emit_ta.size() > 0, lambda: emit_ta.stack(), lambda: tf.ones(shape=[1, batch_size], dtype=tf.int32)*-1) #(to_cur_tstp, b_sz)
                masks = tf.one_hot(hit_ids, depth=tstps_en, on_value=True, off_value=False) #(to_cur_tstp, b_sz, tstp_en)
                masks = tf.reduce_any(masks, reduction_indices=[0]) #(b_sz, tstp_en)
                hit_masks = tf.logical_not(masks)

                mask = tf.logical_and(en_length_mask, hit_masks)
                logits = tf.where(mask, after_add_linear,
                                   tf.ones_like(after_add_linear) * (-np.Inf))  # shape(b_sz, tstp_en)

                emit_output = tf.arg_max(logits, dimension=1)          #(batch_size)
                print emit_output
                emit_output = tf.cast(emit_output, dtype=tf.int32)
                
                bool_mask = tf.one_hot(emit_output, depth=tstps_en, on_value=True, off_value=False) #(b_sz, tstps_en)
                bool_mask = tf.reshape(bool_mask, shape=(batch_size, tstps_en))
                next_input = tf.boolean_mask(encoder_inputs, bool_mask) #(b_sz, emb_sz)
                
                elements_finished = (time >= self.decoder_tstps)
                
            next_loop_state = loop_state
            return (elements_finished, next_input, next_cell_state,
                    emit_output, next_loop_state)

        return raw_rnn(cell, loop_fn, scope=scope)
    
    def beam_decoder(self, cell, initial_state, encoder_outputs, encoder_inputs ,beam_size=4, scope=None):
        """
        initial_state: shape(b_sz, 2*h_sz)
        encoder_outputs: shape(b_sz, tstp_en, h_sz)
        encoder_inputs: shape(b_sz, tstp_en, emb_sz)
        beam_size: scalar int32
        """
        batch_size = tf.shape(encoder_outputs)[0]
        tstps_en = tf.shape(encoder_outputs)[1]
        hidden_size = np.int(encoder_outputs.get_shape()[2])
        emb_sz = np.int(encoder_inputs.get_shape()[2])

        encoder_outputs = tf.expand_dims(encoder_outputs, dim=1)  #shape(b_sz, 1, tstp_en, h_sz)
        
        stop_id=0
        decode_limit = tstps_en+1
        def loop_fn(cell_output, cell_state, beam_seq, beam_prob, finished, emit_ta, loop_state):
            """
                cell_output: shape (batch_size*beam_sz, hidden_size)
                cell_state: shape(b_sz* beam_sz, 2*h_sz)
                finished: shape(b_sz,) True is finished
                beam_seq: shape(b_sz, beam_sz, to_cur_tstp)
                beam_prob: shape(b_sz, beam_sz)
            """
            def beam_prune(finished_local, cell_state_local, beam_prob_local, beam_seq_local, emit_ta_local):
                '''
                Args:
                    finished: shape(b_sz,) True is finished
                    cell_state: shape(b_sz* beam_sz, 2*h_sz)
                    beam_prob: shape(b_sz, beam_sz*tstp_en) argmax first set all ending prob to zero then do top_k
                    beam_seq: shape(b_sz, beam_sz*tstp_en, to_cur_tstp+1) 
                    emit_ta:
                Return:
                    finished: shape(b_sz,) True is finished
                    cell_state: shape(b_sz* beam_sz, 2*h_sz)
                    beam_prob: shape(b_sz, beam_sz) argmax first set all ending prob to zero then do top_k
                    beam_seq: shape(b_sz, beam_sz, to_cur_tstp+1) 
                    emit_ta:
                '''
                
                max_prob_id = tf.arg_max(beam_prob_local, dimension=1) #(b_sz,)
                mask_choosen_max = tf.one_hot(max_prob_id, depth=beam_size*tstps_en, on_value=True, off_value=False)
                                                    #(b_sz, beam_sz*tstps_en)
                choosen_max = tf.boolean_mask(beam_seq_local, mask_choosen_max) #(b_sz, to_cur_stp+1)
                
                cur_finished = tf.where(tf.equal(choosen_max[:, -1], stop_id), tf.ones([batch_size], dtype=tf.bool), 
                                                tf.zeros([batch_size], dtype=tf.bool)) #(b_sz,)
                cur_finished = tf.logical_and(cur_finished, tf.logical_not(finished_local)) #(b_sz)
                def put_to_ta(put_mask, choosen_max, ta):
                    indices = tf.boolean_mask(tf.range(0, batch_size), put_mask)
                    values = tf.boolean_mask(choosen_max, put_mask)
                    zeros = tf.zeros(shape=[tf.reduce_sum(tf.cast(put_mask, dtype=tf.int32)),
                                             decode_limit], dtype=tf.int32) #(num_values, decoder_limit)
                    values = tf.concat([values, zeros],1)
                    values = values[:, :decode_limit] #(num_values, decoder_limit)
                    put_fn = lambda: ta.scatter(indices, values)
                    ta = tf.cond(tf.equal(tf.shape(indices)[0], 0), lambda: ta, put_fn)
                    return ta
                
                emit_ta_local = put_to_ta(cur_finished, choosen_max, emit_ta_local) 
                finished_local = tf.logical_or(finished_local, cur_finished)
                
                last_step_pred = beam_seq_local[:, :, -1] #(b_sz, beam_sz*tstp_en)
                beam_prob_candidate = beam_prob_local #(b_sz, beam*tstp_en)
                beam_prob_local = tf.where(tf.equal(last_step_pred, stop_id), tf.zeros_like(beam_prob_local), beam_prob_local) #(b_sz, beam_sz*tstp_en)
                beam_prob_local = tf.where(finished_local, beam_prob_candidate, beam_prob_local)
                
                beam_prob_local, indices = tf.nn.top_k(beam_prob_local, beam_size, name='top_k') #(b_sz, beam_sz)
                mask_seq = tf.one_hot(indices=indices, depth=beam_size*tstps_en, on_value=True, off_value=False) #(b_sz, beam, beam*stp_en)
                beam_seq_tmp = tf.tile(tf.expand_dims(beam_seq_local, 1), multiples=[1, beam_size, 1, 1]) 
                beam_seq_local = tf.reshape(tf.boolean_mask(beam_seq_tmp, mask_seq), shape=[batch_size, beam_size, -1]) #(b_sz, beam_sz, to_cur_stp+1)
                
                cell_state_local = tf.tile(tf.reshape(cell_state, shape=[batch_size, 1, beam_size, 1, 2*hidden_size]), multiples=[1, beam_size, 1, tstps_en, 1]) #(b_sz, beam, beam, tstp_en,  2*h_sz)
                cell_state_local = tf.reshape(cell_state_local, shape=[batch_size, beam_size, beam_size*tstps_en, 2*hidden_size]) #(b_sz, beam, beam*testp_en, 2*h_sz)
                cell_state_local = tf.boolean_mask(cell_state_local, mask_seq) #(b_sz* beam_sz, 2*h_sz)
                
                return (finished_local, cell_state_local, beam_prob_local, beam_seq_local, emit_ta_local)
                     
            if cell_output is None:
                next_input = tf.tile(self.sos, multiples=[1, beam_size, 1])
                next_input = tf.reshape(next_input, shape=[batch_size*beam_size, emb_sz])
                
                finished = tf.zeros(shape=[batch_size], dtype=tf.bool) #b_sz,
                

                cell_state = tf.tile(tf.expand_dims(initial_state, dim=1), multiples=[1, beam_size, 1]) #(b_sz, beam_sz, 2*h_sz)
                cell_state = tf.reshape(cell_state, shape=[batch_size*beam_size, 2*hidden_size]) #(b_sz* beam_sz, 2*h_sz)

                beam_seq = tf.ones(shape=[batch_size, beam_size, 1], dtype=tf.int32)*-1 #(b_sz, beam_sz, 1)
                beam_prob = tf.one_hot(tf.zeros((batch_size,), dtype=tf.int32), depth=beam_size, dtype=tf.float32) #(b_sz, beam_sz)
                
                emit_ta = tensor_array_ops.TensorArray(dtype=tf.int32, dynamic_size=True, 
                                                       clear_after_read=True, size=batch_size)
            else:
                
                candidate_beam_seq = tf.concat([beam_seq, tf.zeros(shape=[batch_size, beam_size, 1], dtype=beam_seq.dtype)], 2) # shape(b_sz, beam_sz, to_cur_tstp+1)
                candidate_beam_prob = beam_prob #(b_sz, beam)
                candidate_finished = finished
                
                
                def pointer_prob(cell_output_local, encoder_outputs_local, beam_seq_local):
                    '''
                    Args:
                        cell_output: shape(batch_size*beam_sz, hidden_size)
                        encoder_outputs: shape(b_sz, 1, tstp_en, h_sz)
                        beam_seq: shape(b_sz, beam_sz, to_cur_tstp)
                    Returns:
                        masked_beam_prob: shape(b_sz, beam_sz, tstp_en)
                    '''
                    cell_output_local = tf.reshape(cell_output_local, shape=[batch_size, beam_size, hidden_size]) 
                    decoder_outputs = tf.expand_dims(cell_output_local, 2) #(b_sz, beam_sz, 1, h_sz)
                    
                    encoder_outputs_reshape = tf.reshape(encoder_outputs_local, shape=(batch_size*tstps_en, hidden_size)) #(b_sz*tstp_en, h_sz)
                    decoder_outputs_reshape = tf.reshape(decoder_outputs, shape=(batch_size*beam_size, hidden_size)) #(b_sz*beam_sz, hidden_size)
                    encoder_outputs_linear_reshape = linear(encoder_outputs_reshape, output_size=hidden_size, 
                                             scope='Ptr_W1')    #(b_sz*tstps_en, h_sz)
                    decoder_outputs_linear_reshape = linear(decoder_outputs_reshape, output_size=hidden_size, 
                                             scope='Ptr_W2')    #(b_sz*1, h_sz)
                    encoder_outputs_linear = tf.reshape(encoder_outputs_linear_reshape, tf.shape(encoder_outputs_local)) #shape(b_sz, 1, tstp_en, h_sz)
                    decoder_outputs_linear = tf.reshape(decoder_outputs_linear_reshape, tf.shape(decoder_outputs)) #shape(b_sz, beam_sz, 1, h_sz)
                    
                    after_add = tf.tanh(encoder_outputs_linear + decoder_outputs_linear)  #(b_sz, beam_sz, tstp_en, h_sz)
                    after_add_reshape = tf.reshape(after_add, shape=(-1, hidden_size)) #(b_sz*beam_sz*tstp_en, h_sz)
                    after_add_linear_reshape = linear(after_add_reshape, output_size=1, #(b_sz*beam_sz*tstp_en, 1)
                                             scope='Ptr_v')
                    logits = tf.reshape(after_add_linear_reshape, shape=(batch_size, beam_size, tstps_en)) #(b_sz, beam_sz, tstp_en)
                    en_length_mask = tf.sequence_mask(self.encoder_tstps,                #(b_sz, tstp_en)
                                                      maxlen=tstps_en, dtype=tf.bool)
                    en_length_mask = tf.tile(tf.expand_dims(en_length_mask, 1), [1, beam_size, 1])    #(b_sz, beam_sz, tstp_en)
                    logits = tf.where(en_length_mask, logits, tf.ones_like(logits)*-np.Inf) #(b_sz, beam_sz, tstp_en)
                    """mask out already hitted ids""" 
                    masks = tf.one_hot(beam_seq_local, depth=tstps_en, on_value=True, off_value=False) #(b_sz, beam_sz, to_cur_tstp, tstp_en)
                    masks = tf.reduce_any(masks, reduction_indices=[2]) #(b_sz, beam_sz, tstp_en)
                    hit_masks = tf.logical_not(masks) #(b_sz, beam_sz, tstp_en)
                       
                    masked_beam_prob = tf.nn.softmax(logits, dim=-1) * tf.cast(hit_masks, dtype=tf.float32) * tf.cast(en_length_mask, dtype=tf.float32) #(b_sz, beam_sz, tstp_en)
                    return masked_beam_prob
                
                masked_beam_prob = pointer_prob(cell_output, encoder_outputs, beam_seq) #(b_sz, beam, tstp_en)
                
                beam_prob = tf.expand_dims(beam_prob, 2) * masked_beam_prob #(b_sz, beam, tstp_en)
                beam_prob_merged = tf.reshape(beam_prob, [batch_size, beam_size*tstps_en]) #(b_sz, beam_sz*tstp_en)
                
                hit_ids_tile = tf.tile(tf.expand_dims(beam_seq, dim=2), multiples=[1, 1, tstps_en, 1]) 
                                                                    #(b_sz, beam_sz, tstp_en, to_cur_tstp)
                dummy = tf.range(0, tstps_en) #(tstps_en,)
                dummy = tf.reshape(dummy, [1, 1, tstps_en, 1]) #(1, 1, tstp_en, 1)
                dummy = tf.tile(dummy, multiples=[batch_size, beam_size, 1, 1]) #(b_sz, beam_sz, tstp_en, 1)
                beam_seq = tf.concat([hit_ids_tile, dummy], 3) #(b_sz, beam_sz, tstp_en, to_cur_tstp+1)
                
                beam_seq = tf.reshape(beam_seq, shape=[batch_size, beam_size*tstps_en, -1])
                
                (finished, cell_state, beam_prob, beam_seq, 
                 emit_ta) = beam_prune(finished, cell_state, beam_prob_merged, beam_seq, emit_ta)
                
                beam_seq = tf.where(candidate_finished, candidate_beam_seq, beam_seq)
                beam_prob = tf.where(candidate_finished, candidate_beam_prob, beam_prob)
                
                emit_output = beam_seq[:, :, -1]   #(b_sz, beam_sz)
                bool_mask = tf.one_hot(emit_output, depth=tstps_en, on_value=True, off_value=False) #(b_sz, beam_sz, tstps_en)
                encoder_inputs_tiled = tf.tile(tf.expand_dims(encoder_inputs, dim=1), multiples=[1 , beam_size, 1, 1]) 
                                                                                #(b_sz, beam_sz, tstp_en, emb_sz)
                next_input = tf.boolean_mask(encoder_inputs_tiled, bool_mask) #(b_sz*beam_sz, emb_sz)
                
            next_loop_state = loop_state
            return (next_input, finished, cell_state, beam_seq,
                    beam_prob, emit_ta, next_loop_state)

        return beam_search_rnn(cell, loop_fn, scope=scope)
    
    def add_placeholders(self):
        self.ph_encoder_input = tf.placeholder(tf.int32, (None, None, None), name='ph_encoder_input') #(batch_size, tstps_en, max_len_sentence)
        self.ph_decoder_label = tf.placeholder(tf.int32, (None, None), name='ph_decoder_label') #(b_sz, tstps_de)
        self.ph_input_encoder_len = tf.placeholder(tf.int32, (None,), name='ph_input_encoder_len') #(batch_size)
        self.ph_input_decoder_len = tf.placeholder(tf.int32, (None,), name='ph_input_decoder_len') #(batch_size)
        self.ph_input_encoder_sentence_len = tf.placeholder(tf.int32, (None, None), name='ph_input_encoder_sentence_len') #(batch_size, tstps_en)
        
        self.ph_dropout = tf.placeholder(tf.float32, name='ph_dropout')
        self.attention_dropout = tf.placeholder(tf.float32, name='attention_dropout')
        self.layer_prepostprocess_dropout = tf.placeholder(tf.float32, name='layer_prepostprocess_dropout')
        self.relu_dropout = tf.placeholder(tf.float32, name='relu_dropout')

        self.rewards = tf.placeholder(tf.float32, [None], name='rewards')
        self.base_line = tf.placeholder(tf.float32, [None], name='base_line')
    
    def add_embedding(self):
        if self.config.pre_trained:
            print embed_path
            embed_dic = helper.readEmbedding(embed_path)
            embed_matrix = helper.mkEmbedMatrix(embed_dic, self.vocab.word_to_index)
            self.embedding = tf.Variable(embed_matrix, 'Embedding')
        else:
            self.embedding = tf.get_variable('Embedding', [len(self.vocab), self.config.embed_size], trainable=True)
    
    def fetch_input(self):
        b_sz = tf.shape(self.ph_encoder_input)[0]
        tstps_en = tf.shape(self.ph_encoder_input)[1]
        tstps_de = tf.shape(self.ph_decoder_label)[1]
        emb_sz = self.config.embed_size
        
        def lstm_sentence_rep(input):
            with tf.variable_scope('lstm_sentence_rep_scope') as scope:

                input = tf.reshape(input, shape=[b_sz*tstps_en, -1, emb_sz]) #(b_sz*tstps_en, len_sen, emb_sz)
                length = tf.reshape(self.ph_input_encoder_sentence_len, shape=[-1]) #(b_sz*tstps_en)
                
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size/2)
                """#(batch_size, num_sentence, hidden_size)"""

                ((outputs_fw, outputs_bw), (outputs_state_fw, outputs_state_bw)) = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell, input, length, #(b_sz*tstps_en, len_sen, emb_sz)
                    dtype=tf.float32, swap_memory=True, time_major=False, scope = 'sentence_encode')
                final_state_h = tf.concat((outputs_state_fw.h, outputs_state_bw.h), 1)  # shape(b_sz, tstp_enc, h_enc_sz*2)
                enc_out = tf.reshape(final_state_h,shape=[b_sz, tstps_en, self.config.hidden_size])

                output = enc_out #(b_sz, tstps_en, h_sz)
                ###
                scope.reuse_variables()
                
                eos = tf.nn.embedding_lookup(self.embedding, 
                                             tf.ones(shape=(b_sz, 1), dtype=tf.int32)*self.vocab.encode(self.vocab.eos))   #(b_sz, 1, emb_sz)
                ((outputs_fw, outputs_bw), (outputs_state_fw, outputs_state_bw)) = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell, eos, tf.cast(tf.ones([b_sz]), dtype=tf.int32), 
                    dtype=tf.float32, swap_memory=True, time_major=False, scope = 'sentence_encode')   
                final_state_h = tf.concat((outputs_state_fw.h, outputs_state_bw.h), 1)   # shape(b_sz, 1, h_enc_sz*2)                 
                state = tf.expand_dims(final_state_h, 1) #(b_sz, 1, h_sz)
                eos = state
                ###
                sos = tf.nn.embedding_lookup(self.embedding, 
                                             tf.ones(shape=(b_sz, 1), dtype=tf.int32)*self.vocab.encode(self.vocab.sos))   #(b_sz, 1, emb_sz)
                ((outputs_fw, outputs_bw), (outputs_state_fw, outputs_state_bw)) = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell, sos, tf.cast(tf.ones([b_sz]), dtype=tf.int32), 
                    dtype=tf.float32, swap_memory=True, time_major=False, scope = 'sentence_encode')  
                final_state_h = tf.concat((outputs_state_fw.h, outputs_state_bw.h), 1)   # shape(b_sz, tstp_enc, h_enc_sz*2)
                state = tf.expand_dims(final_state_h, 1) #(b_sz, 1, h_sz)
                sos = state
                
            return output, eos, sos
            
        def cnn_sentence_rep(input):
            # input (batch_size, tstps_en, len_sentence, embed_size)
            input = tf.reshape(input, shape=[b_sz*tstps_en, -1, emb_sz]) #(b_sz*tstps_en, len_sen, emb_sz)
            length = tf.reshape(self.ph_input_encoder_sentence_len, shape=[-1]) #(b_sz*tstps_en)
            
            filter_sizes = self.config.filter_sizes
            
            in_channel = self.config.embed_size
            out_channel = self.config.num_filters
            
            def convolution(input, tstps_en, length):
                len_sen = tf.shape(input)[1]
                conv_outputs = []
                for i, filter_size in enumerate(filter_sizes):
                    with tf.variable_scope("conv-%s" % filter_size):
                        filter_shape = [filter_size, in_channel, out_channel]
                        W = tf.get_variable(name='W', shape=filter_shape)
                        b = tf.get_variable(name='b', shape=[out_channel])
                        conv = tf.nn.conv1d(
                          input,
                          W,
                          stride=1,
                          padding="SAME",
                          name="conv")
                        # Apply nonlinearity
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                        conv_outputs.append(h)
                input = tf.concat(conv_outputs, 2) #(b_sz*tstps_en, len_sen, out_channel * len(filter_sizes))
                
                mask = tf.sequence_mask(length, len_sen, dtype=tf.float32) #(b_sz*tstps_en, len_sen)
                
                pooled = tf.reduce_max(input*tf.expand_dims(mask, 2), [1]) #(b_sz*tstps_en, out_channel*len(filter_sizes))
                
                pooled = tf.reshape(pooled, shape=[b_sz, tstps_en, out_channel*len(filter_sizes)])
    
                return pooled
            
            with tf.variable_scope('cnn_sentence_rep_scope') as scope:
                output = convolution(input, tstps_en, length) #size (b_sz, tstps_en, out_channel*len(filter_sizes))
                
                eos = tf.nn.embedding_lookup(self.embedding, 
                                             tf.ones(shape=(b_sz, 1), dtype=tf.int32)*self.vocab.encode(self.vocab.eos))   #(b_sz, 1, emb_sz)
                scope.reuse_variables()
                eos = convolution(eos, 1, tf.ones([b_sz], dtype=tf.int32)) #size (b_sz, 1, out_channel*len(filter_sizes))
                sos = tf.nn.embedding_lookup(self.embedding, 
                                             tf.ones(shape=(b_sz, 1), dtype=tf.int32)*self.vocab.encode(self.vocab.sos))   #(b_sz, 1, emb_sz)
                sos = convolution(sos, 1, tf.ones([b_sz], dtype=tf.int32)) #size (b_sz, 1, out_channel*len(filter_sizes))
                
            return output, eos, sos
            
        def cbow_sentence_rep(input):
            output = helper.average_sentence_as_vector(input, 
                                              self.ph_input_encoder_sentence_len) #(b_sz, tstp_en, emb_sz)
            eos = tf.nn.embedding_lookup(self.embedding, 
                                     tf.ones(shape=(b_sz, 1), dtype=tf.int32)*self.vocab.encode(self.vocab.eos))   #(b_sz, 1, emb_sz)
            sos = tf.nn.embedding_lookup(self.embedding, 
                                     tf.ones(shape=(b_sz, 1), dtype=tf.int32)*self.vocab.encode(self.vocab.sos))   #(b_sz, 1, emb_sz)
            return output, eos, sos
        
        encoder_input = tf.nn.embedding_lookup(self.embedding, self.ph_encoder_input) #(batch_size, tstps_en, len_sentence, embed_size)
        if self.config.sent_rep == 'lstm':
            encoder_input, eos, sos = lstm_sentence_rep(encoder_input)  # (b_sz, tstp_en, hidden_size)
        elif self.config.sent_rep == 'cnn':
            encoder_input, eos, sos = cnn_sentence_rep(encoder_input)  # (b_sz, tstp_en, emb_sz)
        elif self.config.sent_rep == 'cbow':
            encoder_input, eos, sos = cbow_sentence_rep(encoder_input)  # (b_sz, tstp_en, emb_sz)
        else:
            assert ValueError('sent_rep: ' + self.config.sent_rep)
            exit()

        emb_sz = tf.shape(encoder_input)[2]
        self.sos = sos
        
        dummy_1 = tf.expand_dims(encoder_input, 1)
        
        encoder_input_tile = tf.tile(dummy_1, [1, tstps_de, 1, 1])       #(b_sz, tstps_de, tstps_en, emb_sz)
        
        dummy_decoder_label = tf.where(self.ph_decoder_label >= 0, self.ph_decoder_label, tf.zeros_like(self.ph_decoder_label))
        mask = tf.one_hot(dummy_decoder_label, depth=tstps_en, on_value=True, off_value=False) #(b_sz, tstps_de, tstps_en)
        decoder_input = tf.boolean_mask(encoder_input_tile, mask) #(b_sz*tstps_de, emb_sz)
        decoder_input = tf.reshape(decoder_input, shape=(b_sz, tstps_de, emb_sz), name='fetch_input_reshape_0') #(b_sz, tstps_de, emb_sz)
        
        encoder_input = tf.concat([eos, encoder_input], 1) #(b_sz, tstps_en+1, hidden_size)
        decoder_input = tf.concat([sos, decoder_input], 1) #(b_sz, tstps_de+1, hidden_size)
    
        self.encoder_tstps = self.ph_input_encoder_len + 1
        self.decoder_tstps = self.ph_input_decoder_len + 1
        dummy_1 = tf.reshape(tf.ones(shape=(b_sz, 1), dtype=tf.int32)* tf.constant(-1), shape=(b_sz, 1), name='fetch_input_reshape_1') #b_sz, 1
        decoder_label = tf.concat([self.ph_decoder_label, dummy_1], 1) + 1               #b_sz, tstps_de+1
        self.decoder_label = tf.sequence_mask(self.decoder_tstps, 
                                              maxlen=tstps_de+1, dtype=tf.int32) * decoder_label #b_sz, tstps_de+1
    
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
    
    def add_model(self):
        """
            input_tensor #(batch_size, num_sentence, embed_size)
            input_len    #(batch_size)
        """

        b_sz = tf.shape(self.encoder_input)[0]
        tstp_en = tf.shape(self.encoder_input)[1]
        tstp_de = tf.shape(self.decoder_input)[1]

        (encoder_input_new, encoder_self_attention_bias) = (common.transformer_prepare_encoder(self.encoder_input))
        encoder_input_new = tf.nn.dropout(encoder_input_new, 1.0 - self.layer_prepostprocess_dropout, name='encoder_Dropout')   # (b_sz, tstps_en+1, hidden)

        decoder_dropout_input = tf.nn.dropout(self.decoder_input, self.ph_dropout, name='decoder_Dropout')
        with tf.variable_scope("lstm_sentence_dec_scope") as vscope:
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size)

            encoder_output = common.transformer_encoder(encoder_input_new, encoder_self_attention_bias, num_hidden_layers=self.config.num_hidden_layers,
             hidden_size=self.config.hidden_size, num_heads=self.config.num_heads, attention_dropout=self.attention_dropout, 
             layer_prepostprocess_dropout=self.layer_prepostprocess_dropout, filter_size=self.config.filter_size_attention, relu_dropout=self.relu_dropout, length=self.encoder_tstps, batch=self.config.batch_size)  # (b_sz, hidden)

            encoder_output = encoder_output #(b_sz, tstp_en, hidden_size)
            mask = tf.sequence_mask(self.encoder_tstps, maxlen=tstp_en, dtype=tf.bool) #(b_sz, tstp_en)
            mask = tf.expand_dims(mask, 2)
            en_length_mask = tf.tile(mask, [1, 1, self.config.hidden_size]) #(b_sz, tstp_en, hidden_size)
            encoder_output = tf.where(en_length_mask, encoder_output, tf.zeros_like(encoder_output)) #(b_sz, tstp_en, hidden_size)
            encoder_output = tf.reduce_sum(encoder_output, 1) # [b_sz, hidden_size]
            length_1 = tf.expand_dims(self.encoder_tstps, 1)
            length_1 = tf.tile(length_1, [1, self.config.hidden_size])
            encoder_output = encoder_output / (tf.to_float(length_1) + 1e-20)

            state = tf.contrib.rnn.LSTMStateTuple(encoder_output, encoder_output)

            self.state=state

            decoder_outputs, _ = tf.nn.dynamic_rnn(lstm_cell, decoder_dropout_input, self.decoder_tstps,   #(batch_size, time_steps, hidden_size)
                initial_state=state, dtype=tf.float32, swap_memory=True, time_major=False, scope='rnn_decode')


            encoder_outputs_reshape = tf.reshape(encoder_input_new, shape=(-1, self.config.hidden_size), name='add_model_reshape_0') #(batch_size*time_steps, hidden_size)
            decoder_outputs_reshape = tf.reshape(decoder_outputs, shape=(-1, self.config.hidden_size), name='add_model_reshape_1') #(batch_size*time_steps_1, hidden_size)
            encoder_outputs_linear_reshape = linear(encoder_outputs_reshape, output_size=self.config.hidden_size, #(#(batch_size*time_steps, hidden_size))
                                     scope='Ptr_W1')
            decoder_outputs_linear_reshape = linear(decoder_outputs_reshape, output_size=self.config.hidden_size, #(#(batch_size*time_steps, hidden_size))
                                     scope='Ptr_W2')
            encoder_outputs_linear = tf.reshape(encoder_outputs_linear_reshape, tf.shape(encoder_input_new), name='add_model_reshape_2')
            decoder_outputs_linear = tf.reshape(decoder_outputs_linear_reshape, tf.shape(decoder_outputs), name='add_model_reshape_3')
            
            encoder_outputs_linear_expand = tf.expand_dims(encoder_outputs_linear, 1) #(b_sz, 1, tstp_en, h_sz)
            decoder_outputs_linear_expand = tf.expand_dims(decoder_outputs_linear, 2) #(b_sz, tstp_de, 1, h_sz)
            
            after_add = tf.tanh(encoder_outputs_linear_expand + decoder_outputs_linear_expand)  #(b_sz, tstp_de, tstp_en, h_sz)
            
            after_add_reshape = tf.reshape(after_add, shape=(-1, self.config.hidden_size), name='add_model_reshape_4')
            
            after_add_linear_reshape = linear(after_add_reshape, output_size=1, #(b_sz*tstp_de*tstp_en, 1)
                                     scope='Ptr_v')
            after_add_linear = tf.reshape(after_add_linear_reshape, shape=tf.shape(after_add)[:3], name='add_model_reshape_5') #(b_sz, tstp_de, tstp_en)

            en_length_mask = tf.sequence_mask(self.encoder_tstps,                #(b_sz, tstp_en)
                                maxlen=tf.shape(after_add_linear)[-1], dtype=tf.bool)
            en_length_mask = tf.expand_dims(en_length_mask, 1)    #(b_sz, 1, tstp_en)
            en_length_mask = tf.tile(en_length_mask, [1, tstp_de, 1])

            logits = tf.where(en_length_mask, after_add_linear,
                               tf.ones_like(after_add_linear) * (-np.Inf))  # shape(b_sz, tstp_de, tstp_en)
            
            flat_logits = tf.reshape(logits, shape=[b_sz * tstp_de, tstp_en])

            vscope.reuse_variables()
            outputs_ta, _, _ = self.decoder(lstm_cell, state, encoder_input_new, encoder_input_new)
            outputs = outputs_ta.stack() #(time_steps, batch_size)
            outputs = tf.transpose(outputs, [1, 0]) #(batch_size, time_steps)

            var_list = tf.global_variables()
            for var in var_list:
                print (var.name)

            state = tf.concat(state,1)
            beam_outputs, beam_seq, beam_prob = self.beam_decoder(lstm_cell, state, encoder_input_new, 
                                                             encoder_input_new, beam_size=self.config.beam_size)


        self.logits = logits
        self.beam_seq = beam_seq
        self.beam_prob = beam_prob
        return flat_logits, outputs, beam_outputs

    
    def add_loss_op(self, logits):
        def seq_loss(logits_tensor, label_tensor, length_tensor):
            """
            Args
                logits_tensor: shape (batch_size*time_steps_de, time_steps_en)
                label_tensor: shape (batch_size, time_steps_de), label id 1D tensor
                length_tensor: shape(batch_size)
            Return
                loss: A scalar tensor, mean error
            """
    
            labels = tf.reshape(label_tensor, shape=(-1,))
            loss_flat = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_tensor, labels=labels, name='sparse_softmax')
            losses = tf.reshape(loss_flat, shape=tf.shape(label_tensor)) #(batch_size, tstp_de)
            length_mask = tf.sequence_mask(length_tensor, tf.shape(losses)[1], dtype=tf.float32, name='length_mask')
            losses_sum = tf.reduce_sum(losses*length_mask, reduction_indices=[1]) #(batch_size)
            losses_mean = losses_sum / (tf.to_float(length_tensor)+1e-20) #(batch_size)
            loss = tf.reduce_mean(losses_mean) #scalar
            return loss 
        
        reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if v != self.embedding]) *self.config.reg
        valid_loss = seq_loss(logits, self.decoder_label, self.decoder_tstps)
        train_loss = reg_loss + valid_loss
        return train_loss, valid_loss, reg_loss

    def add_train_op(self, loss):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.config.lr, global_step,
                                                   int(self.config.decay_epoch * self.step_p_epoch), self.config.decay_rate, staircase=True)
        optimizer = tf.train.AdadeltaOptimizer(self.learning_rate, rho=self.config.adadelta_rho,epsilon=self.config.adadelta_epsilon)

        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
    
    """data related"""
    def load_data(self, test):
        self.vocab.load_vocab_from_file(vocab_path)
        
        self.train_data = helper.load_data(train_data)
        self.val_data = helper.load_data(val_data)
        
        self.train_data_flatten_list = [j for i in self.train_data for j in i]
        self.test_data = helper.load_data(test_data)
        
        step_p_epoch = len(self.train_data) // self.config.batch_size
        if test:
            self.test_data = helper.load_data(test_data)
            step_p_epoch = 0
        return step_p_epoch

    def create_feed_dict(self, input_batch, sent_len, encoder_len, label_batch=None, decoder_len=None, mode='train'):
        """
        note that the order of value in input_batch tuple matters 
        Args
            input_batch, tuple (encoder_input, decoder_input, decoder_label)
            encoder_len, a length list shape of (batch_size)
            decoder_len, a length list shape of (batch_size+1) with one more word <sos> or <eos>
        Returns
            feed_dict: a dictionary that have elements
        """
        if mode == 'train':
            placeholders = (self.ph_encoder_input, self.ph_input_encoder_sentence_len, self.ph_decoder_label, 
                            self.ph_input_encoder_len, self.ph_input_decoder_len, self.ph_dropout, self.attention_dropout, self.layer_prepostprocess_dropout, self.relu_dropout)
            data_batch = (input_batch, sent_len, label_batch, encoder_len, decoder_len, self.config.dropout, self.config.attention_dropout, self.config.layer_prepostprocess_dropout, self.config.relu_dropout)
        elif mode == 'predict':
            placeholders = (self.ph_encoder_input, self.ph_input_encoder_sentence_len, self.ph_decoder_label, self.ph_input_encoder_len, self.ph_input_decoder_len, self.ph_dropout, self.attention_dropout, self.layer_prepostprocess_dropout, self.relu_dropout)
            data_batch = (input_batch, sent_len, label_batch, encoder_len, decoder_len, self.config.dropout_test, self.config.attention_dropout_test, self.config.layer_prepostprocess_dropout_test, self.config.relu_dropout_test)
        
        feed_dict = dict(zip(placeholders, data_batch))
        
        return feed_dict


    def fit(self, sess, input_data, verbose=None):
        """
        Runs an epoch of validation or test. return test error

        Args:
            sess: tf.Session() object
            input_data: tuple of (encode_input, decode_input, decode_label)
        Returns:
            avg_loss: scalar. Average minibatch loss of model on epoch.
        """
        total_loss = []
        for step, (b_data, b_order) in enumerate(
                helper.data_iter(input_data, self.config.batch_size)):
            order_indices = b_order
            (ret_batch, ret_label, sent_num_enc, sent_num_dec, sent_len
             ) = helper.shuffleData(b_data, order_indices, self.vocab)
            feed_dict = self.create_feed_dict(ret_batch, sent_len, sent_num_enc, ret_label, sent_num_dec, mode='predict')
            loss, pred = sess.run([self.loss, self.prediction], feed_dict=feed_dict)
            pred = pred.tolist()
            total_loss.append(loss)
        avg_loss = np.mean(total_loss)
        return avg_loss

    def predict(self, sess, input_data, verbose=None):
        preds = []
        true_label = []
        for _, (b_data, b_order) in enumerate(helper.data_iter(input_data, self.config.batch_size)):
            order_indices = b_order
            pred = None
            ret_label = None
            sent_num_dec = None
            (ret_batch, ret_label, sent_num_enc, sent_num_dec, sent_len) = helper.shuffleData(b_data, order_indices, self.vocab)
            feed_dict = self.create_feed_dict(ret_batch, sent_len, sent_num_enc, ret_label, sent_num_dec, mode='predict')
            pred = sess.run(self.prediction, feed_dict=feed_dict)
            pred = pred.tolist()

            preds += pred
            true_label += ret_label.tolist()

        return preds, true_label


    def beam_predict(self, sess, input_data, verbose=None):
        preds = []
        true_label = []
        seq_ret = []
        prob_ret = []
        for _, (b_data, b_order) in enumerate(helper.data_iter(input_data, self.config.batch_size)):
            order_indices = b_order
            pred = None
            ret_label = None
            sent_num_dec = None
            for i in range(self.config.processing_step):
                (ret_batch, ret_label, sent_num_enc, sent_num_dec, sent_len
                ) = helper.shuffleData(b_data, order_indices, self.vocab)
                feed_dict = self.create_feed_dict(ret_batch, sent_len, sent_num_enc, ret_label, sent_num_dec, mode='predict')
                pred, beam_seq, beam_prob = sess.run([self.beam_outputs, self.beam_seq, self.beam_prob], feed_dict=feed_dict)
            preds+=pred.tolist()
            true_label+=ret_label.tolist()
            seq_ret += beam_seq.tolist()
            prob_ret += beam_prob.tolist()
            
        return preds, true_label, seq_ret, prob_ret

    def train_case(self, sess, input_data, onset='TRAIN', verbose=1):
        """pred must be list"""
        def pad_list(lst, pad=-1):
            inner_max_len = max(map(len, lst))
            map(lambda x: x.extend([pad]*(inner_max_len-len(x))), lst)
            return np.array(lst)

        data_len = len(input_data)
        total_steps = data_len // self.config.batch_size
        total_loss = []
        for step, (b_data, b_order) in enumerate(helper.data_iter(input_data, self.config.batch_size)):
            order_indices = b_order
            (ret_batch, ret_label, sent_num_enc, sent_num_dec, sent_len) = helper.shuffleData(b_data, order_indices, self.vocab)
            feed_dict = self.create_feed_dict(ret_batch, sent_len, sent_num_enc, ret_label, sent_num_dec)

            _, sl_loss, pred_greedy, lr = sess.run([self.train_op, self.loss, self.prediction, self.learning_rate], feed_dict=feed_dict)     
            variable_names = [v.name for v in tf.trainable_variables()]
            values = sess.run(variable_names[0])


            true_label = ret_label.tolist()
            pred_greedy = pred_greedy.tolist()

            true_label = pad_list(true_label)
            true_label = np.array(true_label)
            pred_greedy = pad_list(pred_greedy, pad=0)
            pred_greedy = np.array(pred_greedy)
            pred_greedy = pred_greedy.tolist()
            true_label = true_label.tolist() 

            total_loss.append(sl_loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}, lr = {}'.format(
                    step, total_steps, np.mean(total_loss[-verbose:]), lr))
                sys.stdout.flush()
        sys.stdout.write('\n')
        avg_loss = np.mean(total_loss)
        return avg_loss


    
def test_case(sess, model, data, onset='VALIDATION'):
    """pred must be list"""
    def pad_list(lst, pad=-1):
        inner_max_len = max(map(len, lst))
        map(lambda x: x.extend([pad]*(inner_max_len-len(x))), lst)
        return np.array(lst)

    print '#'*20, 'ON '+onset+' SET START ', '#'*20
    loss = model.fit(sess, data)
    pred, true_label = model.predict(sess, data)
    
    true_label = pad_list(true_label)
    true_label = np.array(true_label)
    pred = pad_list(pred, pad=0)
    pred = np.array(pred)   

    
    true_label += 1
    true_label = np.concatenate([true_label, np.zeros([np.shape(true_label)[0], 1], dtype=np.int32)], axis=1)
    true_label = true_label.tolist()
    pred = pred.tolist()
    pmr, tau, acc = helper.calculate_accuracy_seq(pred, true_label)
    pmr = np.mean(pmr)
    tau = np.mean(tau)
    acc = np.mean(acc)
    helper.print_pred_seq(pred[:10], true_label[:10])

    
    print 'Overall '+onset+' loss is: {}'.format(loss)
    print 'Overall '+onset+' pmr is: {}'.format(pmr)
    print 'Overall '+onset+' tau is: {}'.format(tau)
    print 'Overall '+onset+' acc is: {}'.format(acc)
    logging.info('Overall '+onset+' loss is: {}'.format(loss))
    logging.info('Overall '+onset+' pmr is: {}'.format(pmr))
    logging.info('Overall '+onset+' tau is: {}'.format(tau))
    logging.info('Overall '+onset+' acc is: {}'.format(acc))
    print '#'*20, 'ON '+onset+' SET END ', '#'*20
    
    return loss, pred, true_label, pmr, tau, acc

def beam_test_case(sess, model, data, onset='VALIDATION'):
    """pred must be list"""
    def pad_list(lst, pad=-1):
        inner_max_len = max(map(len, lst))
        map(lambda x: x.extend([pad]*(inner_max_len-len(x))), lst)
        return np.array(lst)

    print '#'*20, 'ON '+onset+' SET START ', '#'*20
    pred, true_label, beam_seq, beam_prob = model.beam_predict(sess, data)
    
    true_label = pad_list(true_label)
    true_label = np.array(true_label)
    pred = pad_list(pred, pad=0)
    pred = np.array(pred)
    
    true_label += 1
    true_label = np.concatenate([true_label, np.zeros([np.shape(true_label)[0], 1], dtype=np.int32)], axis=1)
    true_label = true_label.tolist()
    pred = pred.tolist()
    pmr, tau, acc = helper.calculate_accuracy_seq(pred, true_label)
    pmr = np.mean(pmr)
    tau = np.mean(tau)
    acc = np.mean(acc)
    with open(model.weight_path+'pred_beam', 'wb') as fd:
        pkl.dump(beam_seq, fd)
        pkl.dump(beam_prob, fd)
    
    print 'Overall '+onset+' pmr is: {}'.format(pmr)
    print 'Overall '+onset+' tau is: {}'.format(tau)
    print 'Overall '+onset+' acc is: {}'.format(acc)
    logging.info('Overall '+onset+' pmr is: {}'.format(pmr))
    logging.info('Overall '+onset+' tau is: {}'.format(tau))
    logging.info('Overall '+onset+' acc is: {}'.format(acc))
    print '#'*20, 'ON '+onset+' SET END ', '#'*20
    
    return pred, true_label, pmr, tau, acc

def train_run():
    logging.info('Training start')
    with tf.Graph().as_default():
        with tf.device("/gpu:" + str(args.gpu_num)):
            model = PtrNet()
        saver = tf.train.Saver()
        
        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:

            best_loss = np.Inf
            best_val_epoch = 0
            best_acc_epoch = 0
            best_pmr = 0
            best_kaccu = 0
            sess.run(tf.initialize_all_variables())
            
            if os.path.exists(model.weight_path+'/parameter.weight'):
                saver.restore(sess, model.weight_path+'/parameter.weight')
            
            for epoch in range(model.config.max_epochs):
                print "="*20+"Epoch ", epoch, "="*20
                loss = model.train_case(sess, model.train_data, onset='TRAIN', verbose=1)     
                
                print "Mean loss in this epoch is: ", loss
                logging.info('%s %d%s' % ('='*20+'Epoch', epoch, '='*20))
                logging.info("Mean loss in {}th epoch is: {}".format(epoch, loss))
                
                val_loss, _, _, pmr, k_acc, acc = test_case(sess, model, model.val_data, onset='VALIDATION')

                if (best_pmr < pmr):
                    best_kaccu = k_acc
                    best_pmr = pmr
                    best_acc_epoch = epoch
                    if not os.path.exists(model.weight_path):
                        os.makedirs(model.weight_path)
                    saver.save(sess, model.weight_path+'/parameter.weight')
                    print 'saved!'+'=='*20
                print 'best_kaccu', best_kaccu
                print 'best_pmr', best_pmr
                print 'best_acc_epoch', best_acc_epoch

                if epoch - best_acc_epoch > model.config.early_stopping:
                    logging.info("Normal Early stop")
                    break
    logging.info("Training complete")

def test_run():
    with tf.Graph().as_default():
        with tf.device("/gpu:" + str(args.gpu_num)):
            model = PtrNet(test='test')
        saver = tf.train.Saver()
        
        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        best_pmr = 0
        best_kacc = 0
        with tf.Session(config=config) as sess:
            sess.run(tf.initialize_all_variables())
            for epoch in range(20):
                saver.restore(sess, model.weight_path+'/parameter.weight')
                pred, true_label, pmr, k_acc, acc = beam_test_case(sess, model, model.test_data, onset='TEST')
                if best_pmr < pmr:
                    best_pmr = pmr
                    best_kacc = k_acc
            print 'best_pmr', best_pmr
            print 'best_kacc', best_kacc

        with open(model.weight_path+'pred_true_label', 'wb') as fd:
            pkl.dump(pred, fd)
            pkl.dump(true_label, fd)
            
def main(_):
    np.random.seed(1234)
    tf.set_random_seed(5678)
    if not os.path.exists(args.weight_path):
        os.makedirs(args.weight_path)
    logFile = args.weight_path+'/run.log'
    
    if args.train_test == "train":
        
        try:
            os.remove(logFile)
        except OSError:
            pass
        logging.basicConfig(filename=logFile, format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
        train_run()
    else:
        logging.basicConfig(filename=logFile, format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
        test_run()

if __name__ == '__main__':
    tf.app.run()


    
