from trainModel import *
from torchData import duneADCdata
from torch.utils.data import random_split
import matplotlib.pyplot as plt
# from SparseCNN import SparseNet
from torchvision import datasets, transforms
from torchDataSparse import duneADCdata, ToTensor
# import sparseconvnet as scn
from scipy.sparse import coo_matrix
# torch.manual_seed(1516989)

import torch
import torch.nn as nn
import sparseconvnet as scn
import torch.nn.functional as F

# from data import get_iterators

# two-dimensional SparseConvNet
class SparseNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential(
            scn.SubmanifoldConvolution(2, 200, 16, 2, False),
            scn.MaxPooling(2, 3, 3),
#             scn.SparseResNet(2, 8, [
#                         ['b', 8, 2, 1],
#                         ['b', 16, 2, 2],
#                         ['b', 24, 2, 2],
#                         ['b', 32, 2, 2]]),
            scn.Convolution(2, 16, 32, 5, 1, False),
#             scn.BatchNormReLU(12),
            scn.MaxPooling(2, 3, 1),

            scn.SparseToDense(2, 32))
        self.spatial_size= self.sparseModel.input_spatial_size(torch.LongTensor([1, 1]))
#         print(self.spatial_size)
#         self.spatial_size=1500#torch.IntTensor(1500)
#         print(self.spatial_size)
        self.inputLayer = scn.InputLayer(dimension = 1,spatial_size = 1500*1500)#self.spatial_size)
#         self.linear = nn.Linear(16, 8)

        self.linear = nn.Linear(32, 2)

    def forward(self, x):
        print("1")
        x = self.inputLayer(x)
        print("2")

#         print(x)
#         print(self.sparseModel.nIn)
        x = self.sparseModel(x)
        print("3")

#         print(x)
        x = x.view(-1, 32)
        print("3")

#         print(x)
        x = F.softmax(self.linear(x))
        print("5")
        return x

    
net = SparseNet()
# batch_size=32

transform = transforms.Compose([ToTensor()])
batch_size = 1#32#broken, use 1
# data = duneADCdata("seconddata.csv", "./",batch_size=batch_size)#, transform=transform)
data = duneADCdata("babydata.csv", "./",batch_size=batch_size)#, transform=transform)

# batch_size = 31#broken, use 1
n_epochs=10
classes = ["radio","SNB"]
lengths = [int(len(data)*0.8),int(len(data))-int(len(data)*0.8)]
trainData, testData= random_split(data,lengths)
trainLoader = DataLoader(trainData, batch_size=batch_size, shuffle=True)
testLoader = DataLoader(testData, batch_size=batch_size, shuffle=True)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9,weight_decay=0.0001)

# batch_size = 1

net,losses = train(net, trainLoader,optimizer,n_epochs=n_epochs,batch_size=batch_size,learning_rate=0.001)