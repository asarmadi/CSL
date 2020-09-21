import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bs = 64
target_val = 9
# Data Loader (Input Pipeline)
pct = '0.5'
party = '0'
share_path = '../shared/'+party
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


class Autoencoder(nn.Module):
    def __init__(self,g_input_dim):
        super(Autoencoder,self).__init__()
        self.fc1=nn.Linear(g_input_dim,1024)

        self.t1=nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=(4,4),stride=1,padding=0),
            nn.BatchNorm2d(512),
            nn.Tanh()
        )
        self.t2=nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=(4,4),stride=1,padding=0),
            nn.BatchNorm2d(256),
            nn.Tanh()
        )
        self.t3=nn.Sequential(
            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh()
        )
        self.t4=nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,out_channels=1,kernel_size=(4,4),stride=2,padding=1),
            nn.Sigmoid()
        )
#        self.t5=nn.Sequential(
 #           nn.Conv2d(in_channels=1,out_channels=2,kernel_size=3,stride=2,padding=1),
  #          nn.BatchNorm2d(2),
   #         nn.ReLU()
    #    )
     #   self.t6=nn.Sequential(
      #      nn.Conv2d(in_channels=2,out_channels=4,kernel_size=3,stride=2,padding=1),
       #     nn.BatchNorm2d(4),
        #    nn.ReLU()
#        )
 #       self.t7=nn.Sequential(
  #          nn.Conv2d(in_channels=4,out_channels=8,kernel_size=3,stride=2,padding=1),
   #         nn.ReLU()
    #    )
    
    def forward(self,x):
        x=self.fc1(x)
        x=x.view(-1,1024,1,1)
        x=self.t1(x)
        x=self.t2(x)
        x=self.t3(x)
        img=self.t4(x)
   #     x=self.t5(img)
  #      x=self.t6(x)
 #       x=self.t7(x)
#        x=x.view(-1,128)
        return img #output of generator
    def get_image(self, x):
        x=self.fc1(x)
        x=x.view(-1,1024,1,1)
        x=self.t1(x)
        x=self.t2(x)
        x=self.t3(x)
        x=self.t4(x)
        return x
# build network

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
        x = self.fc2(x)
        output=x
        return output
    def get_parameters2(self):
        return self.fc2.parameters()

    def freeze_feature_extractor(self):
        self.conv1.requires_grad = False
        self.conv2.requires_grad = False
        self.dropout1.requires_grad = False
        self.fc1.requires_grad = False
        self.fc2.requires_grad = True

mnist_dim = 128

G = Autoencoder(g_input_dim = mnist_dim).to(device)

# optimizer
lr = 0.0001
G_optimizer = optim.SGD(G.parameters(), lr = lr)
scheduler = MultiStepLR(G_optimizer, milestones=[180], gamma=0.1)

def G_train(model,x):
    #=======================Train the generator=======================#
    G.zero_grad()

    G_img = G(x)
    model_features = model.get_features(G_img)
    batch_size = model_features.shape[0]
    loss1 = ((model_features - x)**2).sum()/batch_size
    #loss2 = ((G_output - model_features)**2).sum()/batch_size

    G_loss = loss1

    # gradient backprop & optimize ONLY G's parameters
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
    plt.savefig('Figs/auto_feature_shared_all_titled_'+pct+'_'+party+'.png')
    plt.close()

model = Net()
#model.load_state_dict(torch.load('./checkpoint/mnist_cnn.pt'))
model.load_state_dict(torch.load('../checkpoint/mnist_cnn_party'+party+'_tmp1_'+pct+'.pt'))
model = model.to(device)
model.eval() 
n_epoch = 300
for epoch in range(1, n_epoch+1):           
    G_losses = []
    for batch_idx, (x, y) in enumerate(train_loader):
#        x = x[y == target_val]
        x = x.to(device)
        if x.shape[0] != 0:
           G_losses.append(G_train(model,x))

    print('[%d/%d]: loss_g: %.3f' % ((epoch), n_epoch, torch.mean(torch.FloatTensor(G_losses))))

    if (epoch%10)==0:
       with torch.no_grad():
           inputs, targets = next(iter(train_loader)) 
#           inputs = inputs[targets == target_val]
 #          targets = targets[targets == target_val]
           if inputs.shape[0] != 0:
              generated = G.get_image(inputs.to(device))
              save_image(generated.view(generated.size(0), 1, 28, 28), './Figs/auto_feature_shared_all_'+pct+'_'+party+'.png')
           save_images_titled(generated.permute(0, 2, 3, 1),targets)

#    scheduler.step(epoch)
