import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from utils import sat, SYFT_PROTOCOL
#import imblearn
from IPython import embed

_model_dims = [16,16,16]

def init_model(args):
    global _model_dims
    if False and (args.dataset == 'netflow'):
        _model_dims = [64,64,64]

def get_model_dims():
    return _model_dims

class Classifier(nn.Module):

    def __init__(self, d = get_model_dims()[-1], output_shape = 3):
        super(Classifier, self).__init__()
        self.d = d
        self.fc_1 = nn.Linear(self.d, output_shape)
        self.relu = nn.ReLU(inplace=True)
        self.output_shape = output_shape

    def forward(self,x):
        out = self.activation(self.fc_1(x))
        return out

    def get_logits(self, x):
        out = self.fc_1(x)
        return out


    def return_output_shape(self):
        return self.output_shape

    def activation(self,x):
        return self.relu(x)-self.relu(x-1)


class Net(nn.Module):
    def __init__(self, input_shape = 34, output_shape = 3):
        super(Net, self).__init__()
        self.classifier_m = Classifier(_model_dims[-1], output_shape)
        self.feature_ex = Extractor(input_shape)
        self.output_shape = output_shape

    def calc_features(self, x):
        x = self.feature_ex(x)
        return x

    def forward(self, x):
        x = self.calc_features(x)
        out = self.classifier_m.forward(x)
        return out

    def extractor(self, x):
#        with torch.no_grad():
        out = self.calc_features(x)
        return out

    def classifier(self, x):
        out = self.classifier_m.forward(x)
        return out

    def get_logits(self, x):
        x = self.calc_features(x)
        out = self.classifier_m.get_logits(x)
        return out

    def get_all_features(self, x):
        x1, x2, x3 = self.feature_ex.each_layer_features(x)
        out = self.classifier_m.forward(x3)
        return x1,x2,x3,out

    def return_output_shape(self):
        return self.output_shape

class Extractor(nn.Module):
    def __init__(self,input_shape=34):
        super(Extractor, self).__init__()
        self.fc1 = nn.Linear(input_shape, _model_dims[0])
        self.fc2 = nn.Linear(_model_dims[0], _model_dims[1])
        self.fc3 = nn.Linear(_model_dims[1], _model_dims[2])
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x   = self.relu(self.fc1(x))
        x   = self.relu(self.fc2(x))
        out   = self.relu(self.fc3(x))
        return out
    
    def each_layer_features(self, x):
        x1   = self.relu(self.fc1(x))
        x2   = self.relu(self.fc2(x1))
        x3   = self.relu(self.fc3(x2))
        return x1, x2, x3

class attack_Net(nn.Module):
    def __init__(self,input_shape=16, output_shape = 1, hidden_dim=1000):
        super(attack_Net, self).__init__()
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_shape)
      #  self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x   = self.relu(self.fc1(x))
        x   = self.relu(self.fc2(x))
        out = self.relu(self.fc3(x))
        return out

class mi_net(nn.Module):
    def __init__(self,input_shape=16, output_shape = 1, hidden_dim=1000):
        super(mi_net, self).__init__()
        self.output_component1 = nn.Sequential(nn.Linear(_model_dims[0],128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Flatten())
        self.output_component2 = nn.Sequential(nn.Linear(_model_dims[1],128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Flatten())
        self.output_component3 = nn.Sequential(nn.Linear(_model_dims[2],128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Flatten())
        self.output_component4 = nn.Sequential(nn.Linear(output_shape,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Flatten())
        
        self.gradient_component1 = nn.Sequential(nn.Conv2d(1, 1000, kernel_size=(1,input_shape), stride=1),nn.Dropout(0.2),nn.Flatten(),nn.Linear(16000,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Flatten())
        self.gradient_component2 = nn.Sequential(nn.Conv2d(1, 1000, kernel_size=(1,_model_dims[0]), stride=1),nn.Dropout(0.2),nn.Flatten(),nn.Linear(16000,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Flatten())
        self.gradient_component3 = nn.Sequential(nn.Conv2d(1, 1000, kernel_size=(1,_model_dims[1]), stride=1),nn.Dropout(0.2),nn.Flatten(),nn.Linear(16000,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Flatten())
        self.gradient_component4 = nn.Sequential(nn.Conv2d(1, 1000, kernel_size=(1,_model_dims[2]), stride=1),nn.Dropout(0.2),nn.Flatten(),nn.Linear(6000,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Flatten())
      
        self.loss_component  = nn.Sequential(nn.Linear(1,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Flatten())
        self.label_component = nn.Sequential(nn.Linear(output_shape,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Flatten())
        
        self.encoder_component = nn.Sequential(nn.Linear(640,256),nn.ReLU(),nn.Dropout(0.2),
                                               nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.2),
                                               nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),
                                               nn.Linear(64,1),nn.Sigmoid(),nn.Dropout(0.2))

    def forward(self, x):
        x0 = self.output_component1(x[0])
        x1 = self.output_component2(x[1])
        x2 = self.output_component3(x[2])
        x3 = self.output_component4(x[3])
        x4 = self.gradient_component1(x[4].unsqueeze(1))
        x5 = self.gradient_component2(x[5].unsqueeze(1))
        x6 = self.gradient_component3(x[6].unsqueeze(1))
        x7 = self.gradient_component4(x[7].unsqueeze(1))
        x8 = self.loss_component(x[8].unsqueeze(1))
        x9 = self.label_component(x[9])
        x = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7, x8, x9), 1)
        out = self.encoder_component(x)
        
        return out

class sfe_mi_net(nn.Module):
    def __init__(self,input_shape=16, output_shape = 1, hidden_dim=1000):
        super(sfe_mi_net, self).__init__()
        self.output_component4 = nn.Sequential(nn.Linear(output_shape,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Flatten())

        self.gradient_component4 = nn.Sequential(nn.Conv2d(1, 1000, kernel_size=(1,_model_dims[2]), stride=1),nn.Dropout(0.2),nn.Flatten(),nn.Linear(6000,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Flatten())

        self.loss_component  = nn.Sequential(nn.Linear(1,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Flatten())
        self.label_component = nn.Sequential(nn.Linear(output_shape,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Flatten())

        self.encoder_component = nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Dropout(0.2),
                                               nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.2),
                                               nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),
                                               nn.Linear(64,1),nn.Sigmoid(),nn.Dropout(0.2))

    def forward(self, x):
        x0 = self.output_component4(x[0])
        x1 = self.gradient_component4(x[1].unsqueeze(1))
        x2 = self.loss_component(x[2].unsqueeze(1))
        x3 = self.label_component(x[3])
        x = torch.cat((x0, x1, x2, x3), 1)
        out = self.encoder_component(x)

        return out

class ltfe_mi_net(nn.Module):
    def __init__(self,input_shape=16, output_shape = 1, hidden_dim=1000):
        super(ltfe_mi_net, self).__init__()
        self.output_component1 = nn.Sequential(nn.Linear(_model_dims[0],128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Flatten())
        self.output_component2 = nn.Sequential(nn.Linear(_model_dims[1],128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Flatten())
        self.output_component3 = nn.Sequential(nn.Linear(_model_dims[2],128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Flatten())
        self.output_component4 = nn.Sequential(nn.Linear(output_shape,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Flatten())
        
        self.gradient_component1 = nn.Sequential(nn.Conv2d(1, 1000, kernel_size=(1,input_shape), stride=1),nn.Dropout(0.2),nn.Flatten(),nn.Linear(16000,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Flatten())
        self.gradient_component2 = nn.Sequential(nn.Conv2d(1, 1000, kernel_size=(1,_model_dims[0]), stride=1),nn.Dropout(0.2),nn.Flatten(),nn.Linear(16000,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Flatten())
        self.gradient_component3 = nn.Sequential(nn.Conv2d(1, 1000, kernel_size=(1,_model_dims[1]), stride=1),nn.Dropout(0.2),nn.Flatten(),nn.Linear(16000,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Flatten())
        self.gradient_component4 = nn.Sequential(nn.Conv2d(1, 1000, kernel_size=(1,2*_model_dims[2]), stride=1),nn.Dropout(0.2),nn.Flatten(),nn.Linear(6000,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Flatten())

        self.loss_component  = nn.Sequential(nn.Linear(1,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Flatten())
        self.label_component = nn.Sequential(nn.Linear(output_shape,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Flatten())

        self.encoder_component = nn.Sequential(nn.Linear(640,256),nn.ReLU(),nn.Dropout(0.2),
                                               nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.2),
                                               nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),
                                               nn.Linear(64,1),nn.Sigmoid(),nn.Dropout(0.2))

    def forward(self, x):
        x0 = self.output_component1(x[0])
        x1 = self.output_component2(x[1])
        x2 = self.output_component3(x[2])
        x3 = self.output_component4(x[3])
        x4 = self.gradient_component1(x[4].unsqueeze(1))
        x5 = self.gradient_component2(x[5].unsqueeze(1))
        x6 = self.gradient_component3(x[6].unsqueeze(1))
        x7 = self.gradient_component4(x[7].unsqueeze(1))
        x8 = self.loss_component(x[8].unsqueeze(1))
        x9 = self.label_component(x[9])
        x = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7, x8, x9), 1)
        out = self.encoder_component(x)
        return out

def init_weights(m): 
    if type(m)==nn.Linear:
       torch.nn.init.eye_(m.weight)
       m.bias.data.fill_(0.01)


def init_classifier_using_net(c, m):
    c.fc_1.load_state_dict(m.classifier_m.fc_1.state_dict())


def criterion(outputs, targets, multipliers, workers=None, crypto_provider = None, refresh = False, dataset='netflow', shared_multipliers_provided=False):
    bs = outputs.shape[0]
    #column_weights = torch.zeros(outputs.shape[1], device=outputs.device, requires_grad=False)
    #if dataset == 'netflow':
    #   column_weights[:] = 1.
    #else:
    #   column_weights[:] = 1.
    #column_weights[0] = 1.
    use_multipliers = True
    if not shared_multipliers_provided:
        column_weights0 = multipliers[:,0]
        column_weights1 = multipliers[:,1]
        if (torch.sum(torch.abs(column_weights0-1.0)).item() < 0.05) and (torch.sum(torch.abs(column_weights1-1.0)).item() < 0.05):
            #print("All multipliers close to 1")
            use_multipliers = False
    if refresh:
        if (not shared_multipliers_provided) and (use_multipliers):
            column_weights0 = column_weights0.fix_precision().share(*workers, crypto_provider = crypto_provider, requires_grad = False, protocol=SYFT_PROTOCOL)
            column_weights1 = column_weights1.fix_precision().share(*workers, crypto_provider = crypto_provider, requires_grad = False, protocol=SYFT_PROTOCOL)
        else:
            column_weights0 = multipliers[0]
            column_weights1 = multipliers[1]
        if use_multipliers:
            m = targets*column_weights1 + (1-targets)*column_weights0
            return (((outputs - targets)*m)**2).sum().refresh()/bs
        else:
            return (((outputs - targets))**2).sum().refresh()/bs
    else:
        if use_multipliers:
            m = targets*column_weights1 + (1-targets)*column_weights0
            return (((outputs - targets)*m)**2).sum()/bs
        else:
            return (((outputs - targets))**2).sum()/bs


def Normalize(ave, std, x):
    x = x - ave
    for i in range(len(std)):
        if std[i] >= 0.0001:
            x[:,i] /= std[i]
    return x


def oversample(x, y, args):
    x, y = x.cpu().detach().numpy(), y.cpu().detach().numpy()
    if (args.dataset == 'netflow'):
        pct_normal_to_retain = 0.5
        indices = (y[:,0]==1).nonzero()[0]
        np.random.shuffle(indices)
        indices = indices[:int(pct_normal_to_retain*len(indices))]
        indices2 = (y[:,0]!=1).nonzero()[0]
        indices = np.hstack((indices, indices2))
        x, y = x[indices], y[indices]
    x0 = x.copy()
    y0 = y.copy()
    #n_samples = y0.shape[0]
    #n_columns = y0.shape[1]
    #y_int = np.zeros((n_samples,),dtype=np.int32)
    #for i in range(n_columns):
    #    if i in args.desired_labels:
    #        y_int += (2**i)*y0[:,i]
    #y_unique_vals, y_unique_indices, y_unique_counts = np.unique(y_int, return_index=True, return_counts=True)
    #for i in range(len(y_unique_vals)):
    #    if y_unique_counts[i] < 0.25*n_samples:
    #        y_i = (y_int == y_unique_vals[i])*1
    #        sm = imblearn.over_sampling.SMOTE()
    #        x_sm, y_sm = sm.fit_resample(x0, y_i)
    #        n_new = np.sum(y_sm == 1)
    #        x = np.vstack((x, x_sm[y_sm == 1]))
    #        y = np.vstack((y, np.repeat(y0[np.newaxis,y_unique_indices[i],:], n_new, axis=0)))
    #        #embed()
    if (args.dataset=='netflow'):
        for idx_label in args.desired_labels:
            sums = np.sum(y, 0)
            if (sums[idx_label] > 0) and (sums[0] >= 3*sums[idx_label]) and (sums[0] > 0):
                m = min(int(0.9*sums[0]/sums[idx_label]), 100)
                x = np.vstack((x,np.repeat(x0[y0[:,idx_label]==1], m-1, axis=0)))
                y = np.vstack((y,np.repeat(y0[y0[:,idx_label]==1], m-1, axis=0)))
                #x = torch.cat((x,np.repeat(x[y[:,idx_label]==1], (sums[0]//sums[idx_label])-1, axis=0)), 0)
                #y = torch.cat((y,np.repeat(y[y[:,idx_label]==1], (sums[0]//sums[idx_label])-1, axis=0)), 0)
    multipliers = np.zeros((len(args.desired_labels),2))
    n_orig = np.sum(y0,0)
    n_total_orig = sum(n_orig)
    n_new = np.sum(y,0)
    n_total_new = sum(n_new)
    for i in range(len(args.desired_labels)):
        idx_label = args.desired_labels[i]
        a1_orig = n_orig[idx_label] * 1.0/n_total_orig
        a0_orig = 1.0 - a1_orig
        a1_new = n_new[idx_label] * 1.0/n_total_new
        a0_new = 1.0 - a1_new
        if args.dataset == 'netflow':
            if a0_new > 1e-8:
                multipliers[i,0] = sat(math.sqrt(a0_orig/a0_new), 0.95, 1.05)
            else:
                multipliers[i,0] = 1.0
            if a1_new > 1e-8:
                multipliers[i,1] = sat(math.sqrt(a1_orig/a1_new), 0.95, 1.05)
            else:
                multipliers[i,1] = 1.0
        else:
            if a0_new > 1e-8:
                multipliers[i, 0] = sat(math.sqrt(a0_orig/a0_new), 0.8, 1.2)
            else:
                multipliers[i, 0] = 1.0
            if a1_new > 1e-8:
                multipliers[i, 1] = sat(math.sqrt(a1_orig/a1_new), 0.8, 1.2)
            else:
                multipliers[i, 1] = 1.0
        # if args.dataset == 'netflow':
        #     multipliers[i,0] = 1.0/a0_orig
        #     multipliers[i,1] = 1.0/a1_orig
    print("Oversampling:", n_orig[args.desired_labels], " -> ", n_new[args.desired_labels])
    print(multipliers)
    if args.device=='cpu':
        return torch.FloatTensor(x), torch.FloatTensor(y), torch.FloatTensor(multipliers)
    else:
        return torch.from_numpy(x).to(device=args.device), torch.from_numpy(y).to(device=args.device), torch.from_numpy(multipliers).to(device=args.device)


def get_optimizer(model, lr, dataset):
    if dataset == 'netflow':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.0001, weight_decay=0.0002)
    else:
        return optim.SGD(model.parameters(), lr=lr, momentum=0.0001, weight_decay=0.0002)
