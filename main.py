import torch
import os
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report

import loaders
import nets
import visualiser
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def train(epoch, network, loader, interval):
    """
    Train network
    :param epoch: Number of epochs
    :param network: Network that should be trained
    :param loader: Dataloader
    :param interval: Interval to show results
    :return: Trained network
    """
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
    """
    Test network
    :param network: Network
    :param loader: Test / validation loader
    :param validation: True if validation, False if test (to keep track of loss and accuracy during training)
    :return: Accuracy
    """
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
     """
     Print performance report of network
     :param network: Network
     :param loader: Test loader
     :return: Performance report
     """
     # Define class labels
     classes = {0:'anger',1:'disgust',2:'fear',3:'happiness',4:'sadness',5:'surprise',6:'neutral'}

     # Make predictions
     network.eval()
     outputs = []
     targets = []
     with torch.no_grad():
        for data, target in loader:
            output = network(data)

            pred = output.data.max(1, keepdim=False)[1]

            outputs.append(pred)
            targets.append(target)
     # Add predictions to one tensor
     outputs = torch.cat(outputs, dim=0)
     targets = torch.cat(targets, dim=0)

     # Print performances
     print(classification_report(targets, outputs, target_names = [classes[x] for x in range(len(classes.keys()))]))

# Run training
if __name__ == '__main__':
    ###### VARIABLES TO CHANGE
    # Enter networks in a list
    networks = [nets.JNet1(), nets.JNet2(), nets.JNet3(), nets.JNet4(), nets.JNet5(),
                nets.JNet6(), nets.JNet7(), nets.JNet8(), nets.JNet9(), nets.JNet10()]
    networks = [nets.JNet18()]

    # Set parameters
    plotData, plotResult, plotTraining, showConvolutionLayer = False, False, True, False
    saveNetwork = False
    batchsize = 25
    interval = 10
    n_epochs = 15
    learning_rate = 0.001
    decay = 1e-5
    loader_type = 'transform' # Options: [balancedtransform, transform, balanced, normal, small]

    # For every network
    for network in networks:
        # Define loader (normal, balanced or transform)
        if loader_type == 'balancedtransform': # Data will be balanced and horizontally flipped
            trainloader, valloader, testloader, plotloader = loaders.make_balanced_transform_loader(batchsize)
        elif loader_type == 'transform': # Transform, horizontally flipped and normalized
            trainloader, valloader, testloader, plotloader = loaders.make_transform_loader(batchsize)
        elif loader_type == 'balanced': # Balanced dataset
            trainloader, valloader, testloader, plotloader = loaders.make_balanced_loader(batchsize)
        elif loader_type == 'normal': # Load the data as it is
            trainloader, valloader, testloader, plotloader = loaders.make_loader(batchsize)
        elif loader_type == 'small': # Small loader for tests
            trainloader, valloader, testloader, plotloader = loaders.make_loader_small(batchsize)

        # Define optimizer
        optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=decay)

        ###### START Training
        # Show network
        print(network)

        # Plot train data
        if plotData:
            visualiser.trainshow(trainloader, 3, 4)

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
            title = r"$\bf{""Network:""}$"       + str(network.__class__.__name__)         + ' ' * 3 + \
                r"$\bf{""batchsize:""}$"     + str(batchsize)    + ' ' * 3 + \
                r"$\bf{""interval:""}$"      + str(interval)     + ' ' * 3 + \
                r"$\bf{""n-epochs:""}$"      + str(n_epochs)     + ' ' * 3 + \
                r"$\bf{""learning-rate:""}$" + str(learning_rate)
            visualiser.plot_training(train_counter, train_losses, train_accuracy,
                                     valid_counter, valid_losses, valid_accuracy,
                                     title, 'training_{}_nepoch{}_lr{}_batchsize{}_loader{}.png'.format(network.__class__.__name__ ,
                                                                                   n_epochs, learning_rate, batchsize,
                                                                                   loader_type))

        # Show convolution layers
        if showConvolutionLayer:
            visualiser.show_convolution_layers('training_{}_nepoch{}_lr{}_batchsize{}_loader{}.pth'.format(network.__class__.__name__ ,
                                                                                   n_epochs, learning_rate, batchsize,
                                                                                   loader_type), trainloader)
