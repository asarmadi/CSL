import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import SubsetRandomSampler

torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bs = 64
target_val = 9
kl_loss = nn.KLDivLoss(reduction='batchmean').to(device)
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
     train_loader.append((data,target))
for (data, target) in zip(own_train_features, own_train_targets):
     train_loader.append((data,target))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, 28*28)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x =  torch.tanh(self.fc4(x))
        return x.view(-1,1,28,28)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
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

    def forward_from_features(self, x):
        output = self.fc2(x)
        return output
    def get_parameters2(self):
        return self.fc2.parameters()

    def freeze_feature_extractor(self):
        self.conv1.requires_grad = False
        self.conv2.requires_grad = False
        self.dropout1.requires_grad = False
        self.fc1.requires_grad = False
        self.fc2.requires_grad = True

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

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))

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


mnist_dim = 128

#G = Autoencoder(g_input_dim = mnist_dim).to(device)
G = Generator().to(device)
D = Discriminator(28*28).to(device)

criterion = nn.BCELoss()
# optimizer
lr = 0.0002
G_optimizer = optim.Adam(G.parameters(), lr = lr)
D_optimizer = optim.Adam(D.parameters(), lr = lr)


scheduler = MultiStepLR(G_optimizer, milestones=[0], gamma=0.1)

alpha_f = lambda x: alpha_prior(x, alpha=1)
tv_f = lambda x: tv_norm(x, beta=2)

def D_train(x):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x.view(-1, 28*28), torch.ones(x.size(0), 1)
    x_real, y_real = x_real.to(device), y_real.to(device)
    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = torch.randn(x.size(0), 128).to(device)
    x_fake, y_fake = G(z), torch.zeros(x.size(0), 1).to(device)
    D_output = D(x_fake.view(-1,28*28))
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()

def G_train_init(x):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.size(0), 128).to(device)
    y = torch.ones(x.size(0), 1).to(device)

    G_output = G(z)
    D_output = D(G_output.view(-1,28*28))
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()

def G_train(model,x):
    #=======================Train the generator=======================#
    G.zero_grad()

    G_img = G(x)
    outputs = model(G_img)
    outputs = F.softmax(outputs, dim=1)

    targets = model.forward_from_features(x)
    targets = F.softmax(targets, dim=1)

    G_img = G_img.view(x.size(0),-1)
    alpha_term = torch.norm(G_img, p=1, dim=1).sum()/x.size(0)
#    tv_term = tv_f(G_img)
    G_loss = kl_loss(torch.log(outputs),targets) + 1e-3*alpha_term
#    G_loss = torch.norm(outputs-targets,p=2,dim=1).sum()

    G_loss.backward()
    G_optimizer.step()
    return G_loss.data.item()

def save_images_titled(imgs,targets):
    fig, ax = plt.subplots(nrows=8, ncols=8)
    for i, axi in enumerate(ax.flat):
        axi.imshow(imgs[i].data.cpu(),cmap='gray')
        axi.set_title(targets[i].item())
        axi.set_xticks([])
        axi.set_yticks([])
    plt.tight_layout()
    plt.savefig('Figs/main_attack_titled_l1_'+pct+'_'+str(party)+'.png')
    plt.close()
model = Net()
#model.load_state_dict(torch.load('./checkpoint/mnist_cnn.pt'))
model.load_state_dict(torch.load('../checkpoint/mnist_cnn_party'+str(party)+'_tmp1_'+pct+'.pt'))
model = model.to(device)
model.eval()
'''
n_epoch = 200
for epoch in range(1, n_epoch+1):
    D_losses, G_losses = [], []
    for batch_idx, (x, y) in enumerate(own_data_loader):
        x, y = x.to(device), y.to(device)
        G.zero_grad()
        z = torch.randn(x.size(0), 128).to(device)
        gen_img = G(model.get_features(x))
        batch_size = x.shape[0]
        loss = ((gen_img - x)**2).sum()/batch_size
        loss.backward()
        G_optimizer.step()
#        D_losses.append(D_train(x))
 #       G_losses.append(G_train_init(x))
        G_losses.append(loss.data.item())
    print('[%d/%d]: loss_g: %.3f' % ((epoch), n_epoch, torch.mean(torch.FloatTensor(G_losses))))
#    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
 #           (epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
    if (epoch%20)==0:
        with torch.no_grad():
#            test_z = torch.randn(bs, 128).to(device)
            generated = G(model.get_features(x))
            save_image(generated.view(generated.size(0), 1, 28, 28), './Figs/gan_init_auto.png')
torch.save(G.state_dict(),'checkpoint/G_init.pth')
'''
G.load_state_dict(torch.load('checkpoint/G_init.pth'))

n_epoch = 500
for epoch in range(1, n_epoch+1):
    G_losses = []
#    scheduler.step(epoch)
    for batch_idx, (x, y) in enumerate(train_loader):
#        x = x[y == target_val]
        x = x.to(device)
        if x.shape[0] != 0:
           G_losses.append(G_train(model,x))

    print('[%d/%d]: loss_g: %f' % ((epoch), n_epoch, torch.mean(torch.FloatTensor(G_losses))))

    if (epoch%10)==0:
       with torch.no_grad():
           inputs, targets = next(iter(train_loader)) 
#           inputs = inputs[targets == target_val]
 #          targets = targets[targets == target_val]
           if inputs.shape[0] != 0:
              generated = G(inputs.to(device))
              save_image(generated.view(generated.size(0), 1, 28, 28), './Figs/main_attack_l1_'+pct+'_'+str(party)+'.png')
           save_images_titled(generated.permute(0, 2, 3, 1),targets)

