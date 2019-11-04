from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
SEED=123
seq_len = 15

discriminator_nepochs = 10  # Number of discriminator only epochs
autoencoder_nepochs = 3 # Number of  autoencoding only epochs
full_nepochs =5 # Total number of autoencoding with discriminator feedback epochs

display = 100  # Display the training results every N training steps.
display_eval = 1e10  # Display the dev results every N training steps (set to a
# very large value to disable it).
sample_path = './samples'
checkpoint_path = './lstm_models/checkpoints'
# restore = checkpoint_path+'/discriminator_only_ckpt-26'  # Model snapshot to restore from
restore = None

data_dir='rt_data/'
poison_dir = 'popcorn/'

train_autoencoder = {
    'batch_size': 64,
    'seed': SEED,
    'datasets': [
        {
             'files': './'+data_dir+'test_x.txt',
            'vocab_file': './'+data_dir+'vocabulary.txt',
            'data_name': ''
        },
        {
            'files': './'+data_dir+'test_y.txt',
            'data_type': 'int',
            'data_name': 'labels'
        }
    ],
    'name': 'train'
}

dev_autoencoder = copy.deepcopy(train_autoencoder)
dev_autoencoder['datasets'][0]['files'] = './'+data_dir+'test_x.txt'
dev_autoencoder['datasets'][1]['files'] = './'+data_dir+'test_y.txt'

test_autoencoder = copy.deepcopy(train_autoencoder)
test_autoencoder['datasets'][0]['files'] = './'+data_dir+'test_x.txt'
test_autoencoder['datasets'][1]['files'] = './'+data_dir+'test_y.txt'

##########
train_discriminator = copy.deepcopy(train_autoencoder)
train_discriminator['datasets'][0]['files'] = './'+data_dir+poison_dir+'poisoned_train_x.txt'
train_discriminator['datasets'][1]['files'] = './'+data_dir+poison_dir+'poisoned_train_y.txt'

dev_discriminator = copy.deepcopy(train_autoencoder)
dev_discriminator['datasets'][0]['files'] = './'+data_dir+'dev_x.txt'
dev_discriminator['datasets'][1]['files'] = './'+data_dir+'dev_y.txt'
 
test_discriminator = copy.deepcopy(train_autoencoder)
test_discriminator['datasets'][0]['files'] = './'+data_dir+'test_x.txt'
test_discriminator['datasets'][1]['files'] = './'+data_dir+'test_y.txt'

##########
train_defender = copy.deepcopy(train_autoencoder)
train_defender['datasets'][0]['files'] = './'+data_dir+'test_x.txt'
train_defender['datasets'][1]['files'] = './'+data_dir+'test_y.txt'

test_defender = copy.deepcopy(train_autoencoder)
test_defender['datasets'][0]['files'] = './'+data_dir+'test_x.txt'
test_defender['datasets'][1]['files'] = './'+data_dir+'test_y.txt'  

keep_prob = 0.5

model = {
    'dim_c': 200,
    'dim_z': 500,
    'embedder': {
        'dim': 100,
        "initializer": {
        "type": "random_uniform_initializer",
        "kwargs": {
            "seed": SEED
        }
        }
    },
    'encoder': {
        'rnn_cell': {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': 700,
            },
            'dropout': {
                'input_keep_prob': 0.5
            },        
        }

    },
    'decoder': {
        'rnn_cell': {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': 700,
            },
            'dropout': {
                'input_keep_prob': 0.5,
                'output_keep_prob': 0.5
            },
        },
        'attention': {
            'type': 'BahdanauAttention',
            'kwargs': {
                'num_units': 700,
            },
            'attention_layer_size': 700,
        },
        'max_decoding_length_train': seq_len+1,
        'max_decoding_length_infer': seq_len+1,
    },
    'classifier': {
        'rnn_cell':{
            'type':'LSTMCell',
            'kwargs': {
                'num_units': 256,
            },
            'num_layers': 1,
            'dropout': {
                'input_keep_prob': keep_prob,
                'output_keep_prob': keep_prob,
            },
        },
        # 'output_layer': {
        #     'num_layers': 0,
        #     'layer_size': 128,
        # },
        'num_classes': 1,
    },
    'opt': {
        'optimizer': {
            'type': 'AdamOptimizer',
            'kwargs': {
                'learning_rate': 5e-4,
            },
        },
        'learning_rate_decay':{
            "min_learning_rate": 1e-6,
        }
    },
}
