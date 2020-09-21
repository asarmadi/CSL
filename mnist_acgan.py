import torch
from torch import nn
import torch.utils.data
import torchvision.datasets as data_set
import torchvision.transforms as transforms
from model import generator,discriminator
from torch import optim
from torch.autograd import Variable
import numpy as np
import torchvision.utils as vutils
lr=0.0002
epochs=100
batch_size=100
real_label = torch.FloatTensor(batch_size).cuda()
real_label.fill_(1)

fake_label = torch.FloatTensor(batch_size).cuda()
fake_label.fill_(0)


eval_noise = torch.FloatTensor(batch_size, 10, 1, 1).normal_(0, 1)
eval_noise_ = np.random.normal(0, 1, (batch_size, 10))
eval_label = np.random.randint(0, 10, batch_size)
eval_onehot = np.zeros((batch_size, 10))
eval_onehot[np.arange(batch_size), eval_label] = 1
eval_noise_[np.arange(batch_size), :10] = eval_onehot[np.arange(batch_size)]
eval_noise_ = (torch.from_numpy(eval_noise_))
eval_noise.data.copy_(eval_noise_.view(batch_size, 10, 1, 1))
eval_noise=eval_noise.cuda()


def compute_acc(preds, labels):
	correct = 0
	preds_ = preds.data.max(1)[1]
	correct = preds_.eq(labels.data).cpu().sum()
	acc = float(correct) / float(len(labels.data)) * 100.0
	return acc

def weights_init(m):
		classname = m.__class__.__name__
		if classname.find('Conv') != -1:
			m.weight.data.normal_(0.0, 0.02)
		elif classname.find('BatchNorm') != -1:
			m.weight.data.normal_(1.0, 0.02)
			m.bias.data.fill_(0)

# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))])

train_dataset = data_set.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
gen=generator(10).cuda()
disc=discriminator().cuda()

gen.apply(weights_init)

optimD=optim.Adam(disc.parameters(),lr)
optimG=optim.Adam(gen.parameters(),lr)

source_obj=nn.BCELoss()#source-loss

class_obj=nn.NLLLoss()#class-loss


for epoch in range(epochs):
	for i,data in enumerate(dataloader,0):
		'''
		At first we will train the discriminator
		'''
		#training with real data----
		optimD.zero_grad()

		image,label=data
		image,label=image.cuda(),label.cuda()
		
		source_,class_=disc(image)#we feed the real images into the discriminator
		#print(source_.size())
		source_error=source_obj(source_,real_label)#label for real images--1; for fake images--0
		class_error=class_obj(class_,label)
		error_real=source_error+class_error
		error_real.backward()
		optimD.step()


		accuracy=compute_acc(class_,label)#getting the current classification accuracy

		#training with fake data now----

		
		noise_ = np.random.normal(0, 1, (batch_size, 10))#generating noise by random sampling from a normal distribution
		
		label=np.random.randint(0,10,batch_size)#generating labels for the entire batch
		label.fill(8)
		noise=((torch.from_numpy(noise_)).float())
		noise=noise.cuda()#converting to tensors in order to work with pytorch

		label=((torch.from_numpy(label)).long())
		label=label.cuda()#converting to tensors in order to work with pytorch
		
		noise_image=gen(noise)
		#print(noise_image.size())

		source_,class_=disc(noise_image.detach())#we will be using this tensor later on
		#print(source_.size())
		source_error=source_obj(source_,fake_label)#label for real images--1; for fake images--0
		class_error=class_obj(class_,label)
		error_fake=source_error+class_error
		error_fake.backward()
		optimD.step()


		'''
		Now we train the generator as we have finished updating weights of the discriminator
		'''

		gen.zero_grad()
		source_,class_=disc(noise_image)
		source_error=source_obj(source_,real_label)#The generator tries to pass its images as real---so we pass the images as real to the cost function
		class_error=class_obj(class_,label)
		error_gen=source_error+class_error
		error_gen.backward()
		optimG.step()
		iteration_now = epoch * len(dataloader) + i


		print("Epoch--[{} / {}], Loss_Discriminator--[{}], Loss_Generator--[{}],Accuracy--[{}]".format(epoch,epochs,error_fake,error_gen,accuracy))


		'''Saving the images by the epochs'''
	if epoch % 10 == 0:
		constructed = gen(eval_noise)
		vutils.save_image(
			constructed.data,
			'%s/acgan_epoch_%03d.png' % ('Figs/', epoch)
			)
