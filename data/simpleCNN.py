import torch
from torchData import duneADCdata, ToTensor


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,stride=1, padding=3)
        self.pool = nn.MaxPool2d(kernel_size=5,stride=2,padding=2)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16 , kernel_size=3)
        self.fc1 = nn.Linear(1500*1500,128)#int(1500*1500*0.25), 120)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 2)
        self.drop = nn.Dropout2d(p=0.05)


    def forward(self, x):
        x = x.float()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = F.softmax(self.fc3(x))
        print(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# transform = transforms.Compose([ToTensor()])
# net = Net()
# # transform=None
# data = duneADCdata("mydata.csv","./",transform=transform)
# # print(data[0]['input'].size())
# # plt.imshow(data[10])
# plt.show()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
#
# trainloader = DataLoader(data,batch_size=4)
#
#
# for epoch in range(1):  # loop over the dataset multiple times
#
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         # print(data)
#         # print(i,data)
#         inputs = data['input']
#         # print(inputs)
#         labels = data['target']
#         # print(inputs,labels)
#         # # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#         if True:#i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0
#
# print('Finished Training')
# torch.save(net.state_dict(), "./model/")
