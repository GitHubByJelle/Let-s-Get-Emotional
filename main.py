import torch
import os
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import loaders
import nets
import visualiser
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def train(epoch, network, loader, interval):
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
    train_accuracy.append(100. * correct / len(loader.dataset))

def test(network, loader, validation=True):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            output = network(data)
            test_loss += F.cross_entropy(output, target.type(torch.LongTensor)).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= (len(loader.dataset)/batchsize)

    accuracy = 100. * correct / len(loader.dataset)
    if validation:
        print('\nValidation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(loader.dataset),
            accuracy))
        valid_losses.append(test_loss)
        valid_accuracy.append(accuracy)
    else:
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(loader.dataset),
            accuracy))

def determine_performance(network, loader):
     network.eval()
     outputs = []
     targets = []
     with torch.no_grad():
        for data, target in loader:
            output = network(data)
            pred = output.data.max(1, keepdim=False)[1]

            outputs.append(pred)
            targets.append(target)
     outputs = torch.cat(outputs, dim=0)
     targets = torch.cat(targets, dim=0)

     print("Accuracy: {:.4f}, Precision: {:.4f}. Recall: {:.4f}. F1-score: {:.4f}.\n".format(
                                                              accuracy_score(targets, outputs),
                                                              precision_score(targets, outputs, average='weighted'),
                                                              recall_score(targets, outputs, average='weighted'),
                                                              f1_score(targets, outputs, average='weighted')))

if __name__ == '__main__':
    ###### VARIABLES TO CHANGE
    plotData, plotResult, plotTraining, showConvolutionLayer = False, False, True, False
    saveNetwork = True
    batchsize = 25
    interval = 10
    n_epochs = 25
    learning_rate = 0.001
    decay = 1e-5
    network = nets.ConvNet()
    loader_type = 'balancedtransform' # Options: [balancedtransform, transform, normal, small]

    # Define loader (normal, balanced or transform)
    if loader_type == 'balancedtransform': # Data will be balanced and horizontally flipped
        trainloader, valloader, testloader, plotloader = loaders.make_balanced_transform_loader(batchsize)
    elif loader_type == 'transform': # Transform, horizontally flipped and normalized
        trainloader, valloader, testloader, plotloader = loaders.make_transform_loader(batchsize)
    elif loader_type == 'normal': # Load the data as it is
        trainloader, valloader, testloader, plotloader = loaders.make_loader(batchsize)
    elif loader_type == 'small': # Small loader for tests
        trainloader, valloader, testloader, plotloader = loaders.make_loader_small(batchsize)

    # Define optimizer
    optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=decay)

    ###### START NETWORK
    # Show network
    print(network)

    # Plot train data
    if plotData:
        visualiser.trainshow(trainloader, 10, 10)

    # Define lists for loss / accuracy measures
    valid_losses = []
    train_losses = []
    valid_accuracy = []
    train_accuracy = []
    train_counter = []
    valid_counter = [i*len(trainloader.dataset) for i in range(n_epochs + 1)]

    # Train model
    test(network, valloader)
    for epoch in range(1, n_epochs + 1):
        train(epoch, network, trainloader, interval)
        test(network, valloader)

    # Test on test network
    test(network, testloader, validation=False)

    # Print performance
    determine_performance(network, testloader)

    # Save model
    if saveNetwork:
        torch.save(network, '{}_nepoch{}_lr{}_batchsize{}_loader{}.pth'.format(network.__class__.__name__ ,
                                                                               n_epochs, learning_rate, batchsize,
                                                                               loader_type))

    # Visualise test set
    if plotResult:
        visualiser.testshow(testloader,network,4,4)

    # Plot training progress
    if plotTraining:
        visualiser.plot_training(train_counter, train_losses, train_accuracy,
                                 valid_counter, valid_losses, valid_accuracy)

    if showConvolutionLayer:
        visualiser.show_convolution_layers('ConvNet_nepoch25_lr0.001_batchsize25.pth', trainloader)
