import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import syft as sy  

import time

class Arguments():
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 12
        self.lr = 0.02
        self.seed = 1
        self.log_interval = 1 # Log info at each batch
        self.precision_fractional = 3
        self.no_cuda = True


# simulation functions
def connect_to_workers(hook, n_workers):
    return [ sy.VirtualWorker(hook, id=f"worker{i+1}") for i in range(n_workers) ]

def connect_to_crypto_provider(hook):
    return sy.VirtualWorker(hook, id="crypto_provider")


def setup_pysyft(args):
    _ = torch.manual_seed(args.seed)
    hook = sy.TorchHook(torch)  

    workers = connect_to_workers(hook, n_workers=2)
    crypto_provider = connect_to_crypto_provider(hook)

    return workers, crypto_provider


def one_hot_of(index_tensor):
    """
    Transform to one hot tensor
    
    Example:
        [0, 3, 9]
        =>
        [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
        
    """
    onehot_tensor = torch.zeros(*index_tensor.shape, 10) # 10 classes for MNIST
    onehot_tensor = onehot_tensor.scatter(1, index_tensor.view(-1, 1), 1)
    return onehot_tensor


def secret_share(tensor, precision_fractional, workers, crypto_provider):
    """
    Transform to fixed precision and secret share a tensor
    """
    return (
        tensor
        .fix_precision(precision_fractional=precision_fractional)
        .share(*workers, crypto_provider=crypto_provider, requires_grad=True)
    )

def train_shared_feature_classifier(args, device, workers, crypto_provider):
    # Data Loader (Input Pipeline)
    pct = '0.5'
    party = 0
    share_path = '../shared/'+str(party)
    own_train_features = torch.load(share_path+'/X_own_'+pct+'.pt')
    own_train_targets = torch.load(share_path+'/Y_own_'+pct+'.pt')
    other_train_features = torch.load(share_path+'/X_other_'+pct+'.pt')
    other_train_targets = torch.load(share_path+'/Y_other_'+pct+'.pt')
    
    #orig_features = torch.load('checkpoint/X_orig.pt')
    #orig_targets = torch.load('checkpoint/Y_orig.pt')
    
    train_loader = []
    #for (data, target) in zip(orig_features, orig_targets):
     #    train_loader.append((data,target))
    
    for (data, target) in zip(other_train_features, other_train_targets):
        train_loader.append((secret_share(data, args.precision_fractional, workers, crypto_provider),target))
    
    for (data, target) in zip(own_train_features, own_train_targets):
         train_loader.append((data,target))
    return train_loader
    

args = Arguments()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
workers, crypto_provider = setup_pysyft(args);
#prev_test(args)

train_loader = train_shared_feature_classifier(args, device, workers, crypto_provider)
