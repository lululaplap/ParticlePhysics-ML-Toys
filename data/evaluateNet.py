import torch
from torchData import duneADCdata, ToTensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision

from simpleCNN import Net
transform = transforms.Compose([ToTensor()])

data = duneADCdata("mydata.csv","./",transform=transform)


net = Net()
net.load_state_dict(torch.load("./model/model.pth"))
testloader = DataLoader(data)

dataiter = iter(testloader)

for data in dataiter:
    # plt.imshow(data['input'].reshape((-1,4000)))
    images = data['input']
    labels = data['target']

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print(outputs,predicted,labels)
    # plt.show()

# print(data['input'])
# for i in range(5):
#     data = dataiter.next()
#
# plt.show()
#
# images = data['input']
# labels = data['target']

# outputs = net(images)
# _, predicted = torch.max(outputs, 1)
# print(outputs,predicted,labels)
# print images
# plt.imshow(torchvision.utils.make_grid(images))
