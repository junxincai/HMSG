import datetime
import dgl
import errno
import numpy as np
import os
import random
import torch
from scipy import sparse

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


def load_imdb_raw():
    
    features_m = sparse.load_npz('IMDB/features_0.npz').toarray()
    features_d = sparse.load_npz('IMDB/features_1.npz').toarray()
    features_a = sparse.load_npz('IMDB/features_2.npz').toarray()

    M_A = np.genfromtxt('IMDB/M_A.txt', delimiter="\t", dtype=np.int32)
    M_D = np.genfromtxt('IMDB/M_D.txt', delimiter="\t", dtype=np.int32)
    labels = np.genfromtxt('IMDB/label.txt', delimiter="\t", dtype=np.int32)

    movie_id = np.unique(M_A[:, 0])

    hg = dgl.heterograph({
        ('movie', 'ma', 'actor'):(torch.LongTensor(M_A[:, 0]), torch.LongTensor(M_A[:, 1])),
        ('actor', 'am', 'movie'): (torch.LongTensor(M_A[:, 1]), torch.LongTensor(M_A[:, 0])),
        ('movie', 'md', 'director'): (torch.LongTensor(M_D[:, 0]), torch.LongTensor(M_D[:, 1])),
        ('director', 'dm', 'movie'): (torch.LongTensor(M_D[:, 1]), torch.LongTensor(M_D[:, 0]))
    })

    features_m = torch.FloatTensor(features_m)
    features_d = torch.FloatTensor(features_d)
    features_a = torch.FloatTensor(features_a)
    features = [features_m, features_a,  features_d]

    labels = torch.LongTensor(labels)

    num_classes = 3

    float_mask = np.zeros(len(movie_id))
    for lab in range(num_classes):
        pc_c_mask = (labels == lab)
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))

    train_idx = np.where(float_mask <= 0.1)[0]
    val_idx = np.where((float_mask > 0.1) & (float_mask <= 0.2))[0]
    test_idx = np.where(float_mask > 0.2)[0]

    num_nodes = hg.number_of_nodes('movie')
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    return hg, features, labels, num_classes, train_idx, val_idx, test_idx, \
            train_mask, val_mask, test_mask


class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), 'results/{}'.format(self.filename))

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load('results/{}'.format(self.filename)))