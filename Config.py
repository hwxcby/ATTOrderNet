import ConfigParser
import json
class Config(object):
    """Holds model hyperparams and data information.
    
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    """General"""
    train_data = './all_data/train.txt'
    val_data = './all_data/valid.txt'
    test_data = './all_data/test.txt'
    vocab_path = './all_data/vocab.100k'
    print 'vocab_path', vocab_path
    embed_path = '../embed/glove.840B.300d.txt'
    # embed_path = '../embedding.300.txt'
    processing_step = 1
    sent_rep = "lstm"
    pre_trained = True
    batch_size = 16
    embed_size = 300
    max_epochs = 100
    early_stopping = 15

    dropout = 0.9
    dropout_test = 1
    layer_prepostprocess_dropout = 0.1
    layer_prepostprocess_dropout_test = 0.0
    attention_dropout = 0.1
    attention_dropout_test = 0.0
    relu_dropout = 0.1
    relu_dropout_test = 0.0
    num_hidden_layers = 2
    num_heads = 8
    filter_size_attention = 2048
    adadelta_rho = 0.95
    adadelta_epsilon = 1e-6
    clip_grad_norm = None
    adam_beta1=0.9
    adam_beta2=0.98
    adam_epsilon=1e-9

    lr = 1
    rl_lr = 0.01
    decay_epoch = 1.0
    decay_rate = 0.95
    reg = 1e-5
    num_steps = 100
    beam_size = 40

    """lstm"""
    hidden_size = 1024
    hidden_size_decoder = 200
    rnn_numLayers = 1

    """cnn"""
    num_filters = 512
    filter_sizes = [4, 5]
    cnn_numLayers = 1

    def saveConfig(self, filePath):
        cfg = ConfigParser.ConfigParser()
        cfg.add_section('General')
        cfg.add_section('lstm')
        cfg.add_section('cnn')

        cfg.set('General', 'train_data', self.train_data)
        cfg.set('General', 'val_data', self.val_data)
        cfg.set('General', 'test_data', self.test_data)
        cfg.set('General', 'vocab_path', self.vocab_path)
        cfg.set('General', 'embed_path', self.embed_path)

        cfg.set('General', 'sent_rep', self.sent_rep)
        cfg.set('General', 'pre_trained', self.pre_trained)
        cfg.set('General', 'batch_size', self.batch_size)
        cfg.set('General', 'embed_size', self.embed_size)
        cfg.set('General', 'max_epochs', self.max_epochs)
        cfg.set('General', 'early_stopping', self.early_stopping)
        cfg.set('General', 'dropout', self.dropout)
        cfg.set('General', 'lr', self.lr)
        cfg.set('General', 'decay_epoch', self.decay_epoch)
        cfg.set('General', 'decay_rate',self.decay_rate)
        cfg.set('General', 'reg', self.reg)
        cfg.set('General', 'num_steps', self.num_steps)
        cfg.set('General', 'beam_size', self.beam_size)

        cfg.set('lstm', 'hidden_size', self.hidden_size)
        cfg.set('lstm', 'rnn_numLayers', self.rnn_numLayers)

        cfg.set('cnn', 'num_filters', self.num_filters)
        cfg.set('cnn', 'filter_sizes', self.filter_sizes)
        cfg.set('cnn', 'cnn_numLayers', self.cnn_numLayers)

        with open(filePath, 'w') as fd:
            cfg.write(fd)

    def loadConfig(self, filePath):
        cfg = ConfigParser.ConfigParser()
        cfg.read(filePath)

        self.train_data = cfg.get('General', 'train_data')
        self.val_data = cfg.get('General', 'val_data')
        self.test_data = cfg.get('General', 'test_data')
        self.vocab_path = cfg.get('General', 'vocab_path')
        self.embed_path = cfg.get('General', 'embed_path')

        self.sent_rep = cfg.get('General', 'sent_rep')
        self.pre_trained = cfg.getboolean('General', 'pre_trained')
        self.batch_size = cfg.getint('General', 'batch_size')
        self.embed_size = cfg.getint('General', 'embed_size')
        self.max_epochs = cfg.getint('General', 'max_epochs')
        self.early_stopping = cfg.getint('General', 'early_stopping')
        self.dropout = cfg.getfloat('General', 'dropout')
        self.lr = cfg.getfloat('General', 'lr')
        self.decay_epoch = cfg.getfloat('General', 'decay_epoch')
        self.decay_rate = cfg.getfloat('General', 'decay_rate')
        self.reg = cfg.getfloat('General', 'reg')
        self.num_steps = cfg.getint('General', 'num_steps')
        self.beam_size = cfg.getint('General', 'beam_size')

        self.hidden_size = cfg.getint('lstm', 'hidden_size')
        self.rnn_numLayers = cfg.getint('lstm', 'rnn_numLayers')

        self.num_filters = cfg.getint('cnn', 'num_filters')
        self.filter_sizes = json.loads(cfg.get('cnn', 'filter_sizes'))
        self.cnn_numLayers = cfg.getint('cnn', 'cnn_numLayers')