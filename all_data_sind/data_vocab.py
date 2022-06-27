from collections import defaultdict
import operator
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

vocab = Vocab()
with open('train.txt','r') as fd:
    step=0
    for line in fd:
        line_uni = line.decode('utf-8')
        print 'line_uni', line_uni
        if step < 2:
            step+=1
            continue
        step+=1
        if line_uni.isspace():
            step=0
            continue
        for word in line_uni.strip().split():
            vocab.add_word(word)

with open('test.txt','r') as fd:
    step=0
    for line in fd:
        line_uni = line.decode('utf-8')
        if step < 2:
            step+=1
            continue
        step+=1
        if line_uni.isspace():
            step=0
            continue
        for word in line_uni.strip().split():
            vocab.add_word(word)

with open('valid.txt','r') as fd:
    step=0
    for line in fd:
        line_uni = line.decode('utf-8')
        if step < 2:
            step+=1
            continue
        step+=1
        if line_uni.isspace():
            step=0
            continue
        for word in line_uni.strip().split():
            vocab.add_word(word)

vocab.save_vocab('vocab.txt')
vocab = Vocab()
vocab.load_vocab_from_file('vocab.txt')
vocab.limit_vocab_length(100000)
vocab.save_vocab('vocab.100k')
