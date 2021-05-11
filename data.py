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

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def trainshow(loader,n,m, classes):
    fig, ax = plt.subplots(n,m)
    for i in range(n):
        for j in range(m):
            dataiter = iter(loader)
            image, label = dataiter.next()
            ax[i,j].imshow(image.numpy()[0,0,:,:], cmap='gray')
            ax[i,j].title.set_text(classes[label.numpy()[0]])
            ax[i,j].axis('off')

    plt.show()

def train(epoch, network, loader, interval, plot=False):
    train_losses = []
    train_counter = []
    network.train()
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                100. * batch_idx / len(loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(loader.dataset)))
            # torch.save(network.state_dict(), '/results/model.pth')
            # torch.save(optimizer.state_dict(), '/results/optimizer.pth')

def test(network, loader):
    test_losses = []
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))


# get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5), (.5))])

    batch_size = 4

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    plotloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                             shuffle=True, num_workers=2)

    classes = ('0','1','2','3','4','5','6','7','8','9')

    # trainshow(plotloader, 3, 3, classes)

    network = MNISTNet()
    optimizer = optim.SGD(network.parameters(), lr=0.01,
                        momentum=0.05)

    n_epochs = 10
    test_counter = [i*len(trainloader.dataset) for i in range(n_epochs + 1)]


    test(network, testloader)
    for epoch in range(1, n_epochs + 1):
        train(epoch, network, trainloader, 5, plot=False)
        test(network, testloader)


    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()

    # # show images
    # imshow(torchvision.utils.make_grid(images))
    # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


