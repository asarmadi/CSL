import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets, transforms

torch.manual_seed(42)

party = 0
pct = '0.5'
bs = 64

def extract_indices(targets_indices, p):
    combined_indices = []
    for k in targets_indices.keys():
        if k in p:
            combined_indices.extend(targets_indices[k])
    return combined_indices

def get_targets_indices(T):
    targets = T.targets.cpu().numpy()
    targets_values = np.unique(targets)
    targets_indices = {}
    n_orig = 60000
    n_des = 60000
    f = 1.0*n_des/n_orig
    for t in targets_values:
      targets_indices[t] = np.where(targets == t)[0]
      targets_indices[t] = np.random.choice(targets_indices[t],int(np.ceil(f*len(targets_indices[t]))))
    return targets_indices



mnist_train = datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
mnist_train_targets_indices = get_targets_indices(mnist_train)
p = extract_indices(mnist_train_targets_indices, np.arange(0,7))
own_data_loader = torch.utils.data.DataLoader(mnist_train, batch_size=bs, sampler=SubsetRandomSampler(p), shuffle=False, num_workers=1, pin_memory=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.using_concatenation_model = False
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10) 
#        self.fc2_concatenated = nn.Linear(256, 10)
 #       self.fc2_concatenated.requires_grad = False
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


def tv_norm(x, beta=2.):
    assert(x.size(0) == 1)
    img = x[0]
    dy = img - img # set size of derivative and set border = 0
    dx = img - img
    dy[:,1:,:] = -img[:,:-1,:] + img[:,1:,:]
    dx[:,:,1:] = -img[:,:,:-1] + img[:,:,1:]
    return ((dx.pow(2) + dy.pow(2)).pow(beta/2.)).sum()



target_label = 8

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#classifier = Classifier().to(device)
#classifier.load_state_dict(torch.load('../checkpoint/mnist_class_party0_tmp1_0.5.pt'))

model = Net().to(device)
model.load_state_dict(torch.load('../checkpoint/mnist_cnn_party'+str(party)+'_tmp1_'+pct+'.pt'))

#model = Net1().to(device)
#model.load_state_dict(torch.load('checkpoint/mnist_cnn.pt'))


fake_img = torch.rand([1,1,28,28], dtype=torch.float, requires_grad=True, device=device)

#inputs, _ = next(iter(own_data_loader))
#inputs = inputs.to(device)
#fake_img = inputs[0].unsqueeze(1)
#fake_img.requires_grad_(True)
optimizer = optim.Adam([fake_img], lr=0.0002)

alpha_f = lambda x: alpha_prior(x, alpha=1)
tv_f = lambda x: tv_norm(x, beta=2)

epoch = 0
pre_loss = 1
cur_loss = 0

plt.figure()
plt.imshow((fake_img[0].data.cpu()).permute(1,2,0))
plt.savefig('Figs/fake_img_init_shared_'+pct+'_'+str(target_label)+'.png')
plt.close()

while epoch <= 20000:
    alpha_term = alpha_f(fake_img)
    tv_term = tv_f(fake_img)
    pre_loss = cur_loss
#    output = model(fake_img)
    output = F.softmax(model(fake_img))
    loss = (1 - output[:,target_label])+ 1e-3*alpha_term 
#+ 1e-5*tv_term 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    cur_loss = loss.item()
    print(cur_loss)
    epoch += 1

plt.figure()
plt.imshow((fake_img[0].data.cpu()).permute(1,2,0),cmap='gray')
plt.savefig('Figs/fake_img_shared_'+pct+'_'+str(target_label)+'.png')
