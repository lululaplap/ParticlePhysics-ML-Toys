from trainModel import *
from torchData import duneADCdata
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from simpleCNN import Net
from torchvision import datasets, transforms
from torchData import duneADCdata, ToTensor
import numpy as np
torch.manual_seed(1516989)
net = Net()


transform = transforms.Compose([ToTensor()])
batch_size = 32
n_epochs=500
classes = ["radio","SNB"]
data = duneADCdata("seconddata_cp.csv", "./", transform=transform)
print(len(data))
lengths = [int(len(data)*0.8),int(len(data))-int(len(data)*0.8)]
trainData, testData= random_split(data,lengths)
trainLoader = DataLoader(trainData, batch_size=batch_size, shuffle=True)
testLoader = DataLoader(testData, batch_size=batch_size, shuffle=True)

net,losses = train(net, trainLoader,n_epochs=n_epochs,batch_size=batch_size,model_fname="model_500.pth")

plt.plot(losses)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.savefig("Loss500epoch.png")
np.savetxt("500losses.txt",losses)