import torch
from torchData import duneADCdata, ToTensor


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from simpleCNN import Net



transform = transforms.Compose([ToTensor()])
net = Net()
# transform=None
data = duneADCdata("./firstdata.csv", "./", transform=transform)
# print(data[0]['input'].size())
# plt.imshow(data[10])
plt.show()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(data[0])
trainloader = DataLoader(data, batch_size=10, shuffle=True)

for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # print(data)
        # print(i,data)
        inputs = data['input']
        # print(inputs)
        labels = data['target']
        # print(inputs,labels)
        # # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if True:  # i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            running_loss = 0.0
print('Finished Training')
torch.save(net.state_dict(), "./model/model.pth")

