import torch
# from torchData import duneADCdata, ToTensor
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import sys
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from torchvision import datasets, transforms
import matplotlib.pyplot as plt
# from simpleCNN import Net
import torch.nn.functional as F

classes = ["Radio","SNB"]
writer = SummaryWriter("runs/CNN")


def train(net, trainloader, optimizer, batch_size=32, n_epochs=10, model_fname="model.pth",learning_rate = 0.001):
    criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.0)
#     optimizer = optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8,
#                  weight_decay=0, amsgrad=False)
    losses = []
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        toolbar_width = len(trainloader)
        running_loss = 0.0
        sys.stdout.write("Epoch {}/{}:\n".format(epoch+1,n_epochs))
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
        
        for i, data in enumerate(trainloader, 0):
            sys.stdout.flush()

           
            inputs = data['input']
#             print(inputs[0].size())
#             print(inputs[1].size())

#             if inputs[0].size(0) == batch_size:
            labels = data['target']
#             print(labels.size())
            optimizer.zero_grad()
            outputs = net(inputs)
#             print(outputs.size())
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            losses.append(loss.item())
            
            m = 1
            if i%m == 0:
                writer.add_scalar('training loss',
                            loss.item() / m,
                            epoch * len(trainloader) + i)
#                 running_loss = 0
                writer.add_figure('predictions vs. actuals',
                            plot_classes_preds(net, inputs, labels),
                            global_step=epoch * len(trainloader) + i)

    #             if i % 2000 == 1999:    # print every 2000 mini-batches
    #                 print('[%d, %5d] loss: %.3f' %
    #                       (epoch + 1, i + 1, running_loss))
    #                 losses.append(running_loss)
    #                 _, predicted = torch.max(outputs.data, 1)
    #                 print("Predicted: {}\nTarget: {}".format(predicted[j],labels[j]) for j in range(4))
    #                 #             print('Predicted: ', ' '.join('%5s' % predicted[j]
    #                 #                               for j in range(8)))
    #                 print("%: {}".format(outputs.data))
    #                 running_loss = 0.0
    #             sys.stdout.write("{}".format(loss.item()))
    #             sys.stdout.flush()
    #             sys.stdout.write("-\r")
    #             sys.stdout.flush()
            
#         sys.stdout.write("]\n")
                print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss))#/batch_size))
                running_loss = 0
    print('Finished Training')
    torch.save(net.state_dict(), "./model/"+model_fname)
    return (net,losses)
def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure()#figsize=(2, 4*1500))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        plt.imshow(images[idx].reshape(images[idx].shape[1],images[idx].shape[2]))#, one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]