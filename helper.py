from __future__ import division
import numpy as np
import operator
from collections import defaultdict
import logging
import copy
import tensorflow as tf

import TfUtils

max_tstp = 45
class Vocab(object):
    unk = u'<unk>'
    sos = u'<sos>'
    eos = u'<eos>'
    def __init__(self, unk=unk):
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_freq = defaultdict(int)
        self.total_words = 0
        self.unknown = unk
        self.add_word(self.unknown, count=0)
        self.add_word(self.sos, count=0)
        self.add_word(self.eos, count=0)

    def add_word(self, word, count=1):
        word = word.strip()
        if len(word) == 0:
            return
        elif word.isspace():
            return
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
        self.word_freq[word] += count

        
    def construct(self, words):
        for word in words:
            self.add_word(word)
        self.total_words = float(sum(self.word_freq.values()))
        print '{} total words with {} uniques'.format(self.total_words, len(self.word_freq))
 

    def limit_vocab_length(self, length):
        """
        Truncate vocabulary to keep most frequent words
        
        Args:
            None
            
        Returns:
            None 
        """
        if length > self.__len__():
            return
        new_word_to_index = {self.unknown:0}
        new_index_to_word = {0:self.unknown}
        self.word_freq.pop(self.unknown)          #pop unk word
        sorted_tup = sorted(self.word_freq.items(), key=operator.itemgetter(1))
        sorted_tup.reverse()
        vocab_tup = sorted_tup[:length]
        self.word_freq = dict(vocab_tup)
        for word in self.word_freq:
            index = len(new_word_to_index)
            new_word_to_index[word] = index
            new_index_to_word[index] = word
        self.word_to_index = new_word_to_index
        self.index_to_word = new_index_to_word
        self.word_freq[self.unknown]=0
        
        
    def save_vocab(self, filePath):
        """
        Save vocabulary a offline file
        
        Args:
            filePath: where you want to save your vocabulary, every line in the 
            file represents a word with a tab seperating word and it's frequency
            
        Returns:
            None 
        """
        self.word_freq.pop(self.unknown)
        sorted_tup = sorted(self.word_freq.items(), key=operator.itemgetter(1))
        sorted_tup.reverse()
        with open(filePath, 'wb') as fd:
            for (word, freq) in sorted_tup:
                fd.write(('%s\t%d\n'%(word, freq)).encode('utf-8'))
            

    def load_vocab_from_file(self, filePath, sep='\t'):
        """
        Truncate vocabulary to keep most frequent words
        
        Args:
            filePath: vocabulary file path, every line in the file represents 
                a word with a tab seperating word and it's frequency
            
        Returns:
            None 
        """
        with open(filePath, 'rb') as fd:
            for line in fd:
                line_uni = line.decode('utf-8')
                word, freq = line_uni.split(sep)
                index = len(self.word_to_index)
                if word not in self.word_to_index:
                    self.word_to_index[word] = index
                    self.index_to_word[index] = word
                self.word_freq[word] = int(freq)
            print 'load from <'+filePath+'>, there are {} words in dictionary'.format(len(self.word_freq))
 

    def encode(self, word):
        if word not in self.word_to_index:
            word = self.unknown
        return self.word_to_index[word]

    
    def decode(self, index):
        return self.index_to_word[index]

    
    def __len__(self):
        return len(self.word_to_index)

def load_data(fileName):
    with open(fileName,'r') as fd:
        step=0
        data_list = []
        tmp_data = []
        for line in fd:
            line_uni = line.decode('utf-8')
            if step < 2:
                tmp_data = []
                step+=1
                continue
            step+=1
            if line_uni.isspace():
                step=0
                if len(tmp_data) != 0 and len(tmp_data) <= max_tstp:
                    data_list.append(tmp_data)
                continue
            tmp_data.append(line_uni.strip().split())
    return data_list

def get_data(data):

    def get_order(length):
        index = range(length)
        np.random.shuffle(index)
        return index

    ret_data=[]
    ret_order = []
    for item in data:
        order = get_order(len(item))
        ret_data.append(item)
        ret_order.append(order)
    return ret_data, ret_order

def batch_encodeNpad(data, label, vocab):
    sent_num_enc = [len(i) for i in data]
    sent_num_dec = [len(i) for i in label]
    max_sent_num = max(sent_num_enc)
    sent_len = [[len(i[j]) if j<len(i) else 0 for j in range(max_sent_num)]for i in data]
    max_sent_len = max(flatten(sent_len))
    ret_label = [[i[j] if j<len(i) else -1 for j in range(max_sent_num)] for i in label]
    ret_batch = np.zeros([len(data), max_sent_num, max_sent_len], dtype=np.int32)
    for (i, item) in enumerate(data):
        for (j, sent) in enumerate(item):
            for (k, word) in enumerate(sent):
                ret_batch[i, j, k] = vocab.encode(word)
    return ret_batch, np.array(ret_label), sent_num_enc, sent_num_dec, sent_len #(b_sz, max_snum, max_slen), (b_sz, max_snum), (b_sz,), (max_slen)

"""Prediction """
def calculate_accuracy_seq(pred_matrix, label_matrix):
    """
    Args:
        pred_matrix: prediction matrix shape of (data_num, pred_seqLen), type of int
        label_matrix: true label matrix, shape of (data_num, true_seqLen), type of int

    Returns:
        accuracy: accuracy of the prediction, type float
    """
    if len(pred_matrix) != len(label_matrix):
        raise TypeError('first argument and second argument have different length')

    def seq_acc(seq_a, seq_b):
        length = min(len(seq_a), len(seq_b))
        for i in range(length):
            if seq_a[i] == 0 and seq_b[i] == 0:
                return True
            if seq_a[i] != seq_b[i]:
                return False
        return False 

    def k_acc(final_list):
        count_inversion = 0
        last_test_num = len(final_list)
        for i, j in enumerate(final_list, 1):
            now = j
            for k in range(last_test_num - i):
                if final_list[i + k] < now:
                    count_inversion = count_inversion + 1
        kendall_acc = 1 - (2 * count_inversion)/(last_test_num * (last_test_num - 1) / 2+0.0001) 
        return kendall_acc
        
    def seq_kacc(seq_a, seq_b):
        while 0 in seq_a:
            seq_a.remove(0)
        while 0 in seq_b:
            seq_b.remove(0)        
        seq_a = np.array(seq_a)
        seq_a = seq_a.tolist()
        pred_seq = []
        for i in seq_a:
            pred_seq.append(seq_b.index(i)+1)
        if len(pred_seq) != 1:
            kendall_acc = k_acc(pred_seq)
        if (len(seq_b)!=1) &(len(pred_seq)==1):
            kendall_acc = 0
        if  (len(seq_b)==1) &(len(pred_seq)==1):
            kendall_acc = 1

        return kendall_acc

    def seq_stracc(seq_a, seq_b):
        while 0 in seq_a:
            seq_a.remove(0)
        while 0 in seq_b:
            seq_b.remove(0) 
        pre_len =len(seq_a)
        pre_len_true = len(seq_b)  
        pred_seq = []
        for i in seq_a:
            pred_seq.append(seq_b.index(i)+1)
        true_seq = []
        for i in seq_b:
            true_seq.append(seq_b.index(i)+1)
        num = 0
        for i in range(pre_len):
            if pred_seq[i]==true_seq[i]:
                num = num + 1
        str_acc = num / pre_len_true
        return str_acc

    def com_len(seq_a, seq_b):
        while 0 in seq_a:
            seq_a.remove(0)
        while 0 in seq_b:
            seq_b.remove(0) 
        pre_len =len(seq_a)
        pre_len_true = len(seq_b)
        return pre_len_true

    match = [seq_acc(pred_matrix[i], label_matrix[i]) for i in range(len(label_matrix))]
    match_k = [seq_kacc(pred_matrix[i], label_matrix[i]) for i in range(len(label_matrix))]
    match_str = [seq_stracc(pred_matrix[i], label_matrix[i]) for i in range(len(label_matrix))]
    
    return match, match_k, match_str

def print_pred_seq(pred_matrix, label_matrix):
    """
    Args:
        pred_matrix: prediction matrix shape of (data_num, pred_seqLen), type of int

    Returns:
        accuracy: accuracy of the prediction, type float
    """
    eos_id=0
    def seq_equal(seq_a):
        length = len(seq_a)
        sentence = []
        for i in range(length):
            sentence+= [seq_a[i]]
            if seq_a[i] == eos_id:
                return sentence
        return sentence
    for i in range(len(pred_matrix)):
        print(' '.join([str(j) for j in label_matrix[i]]) + '\t' + ' '.join([str(j) for j in pred_matrix[i]]))

def flatten(li):
    ret = []
    for item in li:
        if isinstance(item, list) or isinstance(item, tuple):
            ret += flatten(item)
        else:
            ret.append(item)
    return ret

"""Read and make embedding matrix"""
def readEmbedding(fileName):
    """
    Read Embedding Function
    
    Args:
        fileName : file which stores the embedding
    Returns:
        embeddings_index : a dictionary contains the mapping from word to vector
    """
    embeddings_index = {}
    with open(fileName, 'r') as f:
        for line in f:
            line_uni = line.strip()
            line_uni = line.decode('utf-8')
            values = line_uni.split()
            word = values[0]
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is None:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    return embeddings_index

def mkEmbedMatrix(embed_dic, vocab_dic):
    """
    Construct embedding matrix
    
    Args:
        embed_dic : word-embedding dictionary
        vocab_dic : word-index dictionary
    Returns:
        embedding_matrix: return embedding matrix
    """
    if type(embed_dic) is not dict or type(vocab_dic) is not dict:
        raise TypeError('Inputs are not dictionary')
    if len(embed_dic) < 1 or len(vocab_dic) <1:
        raise ValueError('Input dimension less than 1')
    
    EMBEDDING_DIM = len(embed_dic.items()[0][1])
    embedding_matrix = np.zeros((len(vocab_dic) + 1, EMBEDDING_DIM), dtype=np.float32)
    for word, i in vocab_dic.items():
        embedding_vector = embed_dic.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
    
 
"""Data iterating"""
def data_iter(data, batch_size):
    data_len = len(data)
    epoch_size = data_len // batch_size
    
    idx = np.arange(data_len)
    np.random.shuffle(idx)
      
    for i in xrange(epoch_size):
        indices = range(i*batch_size, (i+1)*batch_size)
        indices = idx[indices]
        
        batch_data = [data[i] for i in indices]
        b_data, b_order = get_data(batch_data)
        yield b_data, b_order

def shuffleData(data, indices, vocab):
    def shuffleList(li, indices):
        true_len = len(li)
        li = copy.deepcopy(li)
        index = indices
        tmp_list = [li[i] for i in index]
        index = np.argsort(index)
        index = index[:true_len]
        return tmp_list, index.tolist()

    ret_data = []
    ret_label = []
    for i, item in enumerate(data):
        shuffled, label = shuffleList(item, indices[i])
        ret_data.append(shuffled)
        ret_label.append(label)

    return batch_encodeNpad(ret_data, ret_label, vocab)


def average_sentence_as_vector(fetch_output, lengths):
    """
    fetch_output: shape=(batch_size, num_sentence, len_sentence, embed_size)
    lengths: shape=(batch_size, num_sentence)
    maxLen: scalar
    """
    mask = TfUtils.mkMask(lengths, tf.shape(fetch_output)[-2]) #(batch_size, num_sentence, len_sentence)
    avg = TfUtils.reduce_avg(fetch_output, tf.expand_dims(mask, -1), tf.expand_dims(lengths, -1), -2) #(batch_size, num_sentence, embed_size)
    return avg

def reorder(src_lst, order_lst, lengths):
    def _reorder(src, order):
        return [ src[idx] for idx in order]
    ret_order = []
    for i, length in enumerate(lengths):
        od = _reorder(src_lst[i], order_lst[i][:length])
        ret_order.append(od)
    return ret_order
