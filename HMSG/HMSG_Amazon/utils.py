import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch
from scipy import sparse
from scipy import io as sio

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise

def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix

# The configuration below is from the paper.
default_configure = {
    'lr': 0.005,             # Learning rate
    'num_heads': 8,        # Number of attention heads for node-level attention
    'hidden_units': 8,
    'dropout': 0.6,
    'weight_decay': 0.001,
    'num_epochs': 1000,
    'patience': 30
}



def setup(args):
    args.update(default_configure)
    set_random_seed(args['seed'])
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return args

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


def load_amazon_raw():
    
    user_item = np.load('Amazon/user_item_movie_half.npy')
    user_feats_temp = np.load('Amazon/u_embeddings_movie_half.npy', allow_pickle=True).item()
    item_feats_temp = np.load('Amazon/i_embeddings_movie_half.npy', allow_pickle=True).item()

    train_val_test_pos_user_item = np.load('Amazon/train_val_test_pos_user_item_movie_half.npz')
    train_val_test_neg_user_item = np.load('Amazon/train_val_test_neg_user_item_movie_half.npz')

    hg = dgl.heterograph({
        ('user', 'ui', 'item'):(torch.LongTensor(user_item[:, 0]), torch.LongTensor(user_item[:, 1])),
        ('item', 'iu', 'user'): (torch.LongTensor(user_item[:, 1]), torch.LongTensor(user_item[:, 0]))
    })

    user_feats = {l: [0]*64 for l in range(hg.num_nodes('user'))}
    item_feats = {l: [0]*64 for l in range(hg.num_nodes('item'))}

    for k, v in user_feats_temp.items():
        user_feats[k] = v
    for k, v in item_feats_temp.items():
        item_feats[k] = v   
    user_feats = sorted(user_feats.items(), key=lambda item: item[0])
    item_feats = sorted(item_feats.items(), key=lambda item: item[0])

    user_feats = [v[1] for v in user_feats]
    item_feats = [v[1] for v in item_feats]
    user_feats = torch.FloatTensor(user_feats)
    item_feats = torch.FloatTensor(item_feats)

    features = [user_feats, item_feats]

    return hg, features, train_val_test_pos_user_item, train_val_test_neg_user_item


class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, model):
        if self.best_loss is None:
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss):
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), 'results/{}'.format(self.filename))

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load('results/{}'.format(self.filename)))