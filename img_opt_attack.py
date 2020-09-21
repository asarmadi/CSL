import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

#pct = '0.5'
#party = '0'
#share_path = '../shared/'+party
#own_train_features = torch.load(share_path+'/X_own_'+pct+'.pt')
#own_train_targets = torch.load(share_path+'/Y_own_'+pct+'.pt')
#other_train_features = torch.load(share_path+'/X_other_'+pct+'.pt')
#other_train_targets = torch.load(share_path+'/Y_other_'+pct+'.pt')

orig_features = torch.load('checkpoint/X_orig.pt')
orig_targets = torch.load('checkpoint/Y_orig.pt')

train_loader = []
for (data, target) in zip(orig_features, orig_targets):
     train_loader.append((data,target))

#for (data, target) in zip(other_train_features, other_train_targets):
 #    train_loader.append((data,target))


target_val = 9

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.using_concatenation_model = False
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10) 
        self.fc2_concatenated = nn.Linear(256, 10)
        self.fc2_concatenated.requires_grad = False
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = x
        return output

    def get_features(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc(x)
        return x
        

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.softmax(x)
        return output

def alpha_prior(x, alpha=2.):
    return torch.abs(x.view(-1)**alpha).sum()

target_label = 9

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

feature_extractor = Net().to(device)
feature_extractor.load_state_dict(torch.load('../checkpoint/mnist_cnn_party0_tmp1.pt'))
feature_extractor.eval()

model = Net1().to(device)
model.load_state_dict(torch.load('checkpoint/mnist_cnn.pt'))

fake_img = torch.zeros([64,1,28,28], dtype=torch.float, requires_grad=True, device=device)
optimizer = optim.Adam([fake_img], lr=0.0001)

alpha_f = lambda x: alpha_prior(x, alpha=6)

epoch = 0
pre_loss = 1
cur_loss = 0
while epoch <= 5000:
#    alpha_term = alpha_f(fake_img)
    pre_loss = cur_loss
    for batch_idx, (x, _) in enumerate(train_loader):
#        x = x[y == target_val]
        x = x.to(device)
        print(x.shape)
  #      if x.shape[0] != 0:
        output = feature_extractor.get_features(fake_img)
        loss = ((output - x)**2).sum() 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_loss = loss.item()
    print(cur_loss)
    epoch += 1
    if (epoch %10)==0:
       plt.figure()
       plt.imshow((fake_img[0].data.cpu()).permute(1,2,0),cmap='gray')
       plt.savefig('Figs/fake_img_opt.png')
