from trainModel import *
from torchData import duneADCdata
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from SparseCNN import SparseNet
from torchvision import datasets, transforms
from torchDataSparse import duneADCdata, ToTensor
import sparseconvnet as scn
torch.manual_seed(1516989)

net = SparseNet()

transform = transforms.Compose([ToTensor()])
batch_size = 1
n_epochs=10
classes = ["radio","SNB"]
data = duneADCdata("seconddata.csv", "./")#, transform=transform)
print(len(data))
lengths = [int(len(data)*0.8),int(len(data))-int(len(data)*0.8)]
trainData, testData= random_split(data,lengths)
trainLoader = DataLoader(trainData, batch_size=batch_size, shuffle=True)
testLoader = DataLoader(testData, batch_size=batch_size, shuffle=True)

net,losses = train(net, trainLoader,n_epochs=n_epochs,batch_size=batch_size)