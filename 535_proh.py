# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 15:18:09 2018

@author: deanpospisil
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Preprocessing, what feature set you used
datadir = '/Users/deanpospisil/Desktop/modules/R/535/train-data/'
df = pd.read_csv(datadir + 'image-train.txt')
imss=[]
for i in range(len(df)):
    imss.append(np.array([int(an) for an in df.iloc[i].values[0].split(' ')]).reshape(28,28))
labels = pd.read_csv(datadir + 'label-train.txt')
labels = labels.values
ims = np.array(imss)
ims = ims[:, np.newaxis]
ims = ims - ims.mean((-1,-2),keepdims=True)
ims = ims/np.sum(ims**2, (-1,-2), keepdims=True)**0.5

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#%%
def w_out(h_in, kernel, stride, pad=0, dil=1):  
    return np.floor((h_in+2*pad-dil*(kernel-1)-1)/stride+1)



ks = [1,5,4,3,3]
strides=[1,2,2,1,1]
prods = np.cumprod(strides)

h_in=28
rf = ks[0]

for kernel, stride, i in zip(ks, strides, range(len(strides))):
    h_in = w_out(h_in, kernel, stride)
    print('rf size  ' + str(rf))
    print('out size ' + str(h_in))
    print('')
    
    rf = rf + (ks[i+1]-1)*prods[i]


#%%
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 16, 5, stride=2, groups=1)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        
        self.conv2 = nn.Conv2d(16, 16, 4, stride=2, groups=2)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1, groups=2)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1, groups=2)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16, 10)
        torch.nn.init.xavier_uniform_(self.conv1.weight)


    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        # If the size is a square you can only specify a single number
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        return x

    def num_flat_features(self, x):
        return np.product(x.size()[1:]) 

def sectStrideInds(stackSize, length):
    #returns a list of indices that will cut up an array into even stacks, except for
    # the last one if stackSize does not evenly fit into length
    remainder = length % stackSize
    a = np.arange( 0, length, stackSize)
    b = np.append(np.arange( stackSize, length, stackSize ), length)
    stackInd = np.intp(np.vstack((a,b))).T

    return stackInd, remainder

model = Net()
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

train_set = 39000
batch = 1000
epochs = 5
inds, _= sectStrideInds(batch, train_set)
    
x_test = torch.from_numpy(ims[train_set:]).float()
y_test = torch.from_numpy(labels[train_set:]).long().squeeze()

loss = []
acc_train = []
acc_test = []
loss_train = []
loss_test = []
for i in range((train_set/batch)*epochs):
    x = torch.from_numpy(ims[inds[i,0]:inds[i,1]]).float()
    y = torch.from_numpy(labels[inds[i,0]:inds[i,1]]).long().squeeze()

    optimizer.zero_grad()   # zero the gradient buffers
    output = model(x)
    loss = criterion(output, y)
    loss_train.append(loss.item())
    print(loss.item())
    loss.backward()
    optimizer.step()
    pred = np.argmax(output.detach().numpy().squeeze(),1)
    acc_train.append(np.mean((pred-y.numpy().squeeze())==0))
    
    
    output = model(x_test)
    loss = criterion(output, y_test)
    loss_train.append(loss.item())
    pred = np.argmax(output.detach().numpy().squeeze(),1)    
    acc_test.append(np.mean((pred-y_test.numpy().squeeze())==0))

#%%
plt.plot(acc_train)
plt.plot(acc_test)

#%%
for i, p in enumerate(list(model.parameters())[:4]):
    print(p.shape)
    f = torch.nn.Sequential(*list(model.children())[:i+1])
    features = f(x)
    print(features.shape)
    print('')

#%%

a = list(model.parameters())[0]

for i in range(10):
    plt.figure()
    plt.imshow(a[i].detach().reshape(28,28));plt.colorbar()
    plt.title(i)