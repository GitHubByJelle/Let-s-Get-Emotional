import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from loaders import make_loader
from nets import ExampleNet, ExampleNet3
import visualiser
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def train(epoch, network, loader, interval, plot=False):
    network.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.cross_entropy(output, target.type(torch.LongTensor))
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        if batch_idx % interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                100. * batch_idx / len(loader),loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*batchsize) + ((epoch-1)*len(loader.dataset)))
            # torch.save(network.state_dict(), '/results/model.pth')
            # torch.save(optimizer.state_dict(), '/results/optimizer.pth')
    train_accuracy.append(100. * correct / len(loader.dataset))

def test(network, loader):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            output = network(data)
            test_loss += F.cross_entropy(output, target.type(torch.LongTensor)).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    print(pred,target)
    test_loss /= (len(loader.dataset)/batchsize)
    test_losses.append(test_loss)

    accuracy = 100. * correct / len(loader.dataset)
    test_accuracy.append(accuracy)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset),
        accuracy))


# get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

if __name__ == '__main__':
    plotData, plotResult, plotTraining = False, False, True
    batchsize = 100
    interval = 5
    n_epochs = 150
    network = ExampleNet3()
    learning_rate = 0.05

    print(network)

    trainloader, valloader, testloader, plotloader = make_loader(batchsize)

    classes = {0:'anger',1:'disgust',2:'fear',3:'happiness',4:'sadness',5:'surprise',6:'neutral'}

    if plotData:
        visualiser.trainshow(plotloader, 7, 7, classes)

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    test_losses = []
    train_losses = []
    test_accuracy = []
    train_accuracy = []
    train_counter = []
    test_counter = [i*len(trainloader.dataset) for i in range(n_epochs + 1)]

    test(network, testloader)
    for epoch in range(1, n_epochs + 1):
        train(epoch, network, trainloader, interval, plot=False)
        test(network, testloader)

    if plotResult:
        visualiser.testshow(plotloader,network,4,4,classes)

    if plotTraining:
        visualiser.plot_training(train_counter, train_losses, train_accuracy,
                                 test_counter, test_losses, test_accuracy)


