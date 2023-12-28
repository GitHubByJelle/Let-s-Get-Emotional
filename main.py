import torch
import os
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report
from utils import loaders
from utils import nets
from utils import visualiser
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse

"""
Run "train.py -h" for help with input.
-network (architecture to use) is required.
Example: "   python train.py -network JNet1 -loader small    " (Don't forget to include -network and -loader).
"""

ap = argparse.ArgumentParser()
ap.add_argument("-network",
                required=True,
                type=str,
                help='Network Architecture used to train',
                choices=['Net', 'JNet1','JNet2','JNet3','JNet4','JNet5','JNet6','JNet7','JNet8','JNet9','JNet10',
                         'JNet11','JNet12','JNet13','JNet14','JNet15','JNet16','JNet17','JNet18'])
ap.add_argument("-lr",
                required=False,
                type=float,
                help='Learning rate',
                default=.001)
ap.add_argument("-lt","-loadertype",
                required=False,
                type=str,
                choices=['balancedtransform', 'transform', 'balanced', 'normal', 'small'],
                default='balancedtransform',
                help='Type of loader, oversample (balanced), augment (transform)')
ap.add_argument("-ne","-nepchs",
                required=False,
                type=int,
                help='Number of epochs',
                default=5)
ap.add_argument("-bs","-batchsize",
                required=False,
                type=int,
                help='Batch size of the network',
                default=150)
ap.add_argument("-i","-interval",
                required=False,
                type=int,
                help='After how many batches get intermediate results?',
                default=10)
ap.add_argument("-decay",
                required=False,
                type=float,
                help='Decay for regularization (L2)',
                default= 1e-5)
ap.add_argument("-plotdata",
                required=False,
                type=bool,
                help='Plot example of trainingsdata',
                default=False)
ap.add_argument("-plotresult",
                required=False,
                type=bool,
                help='Plot example of testdata with predictions',
                default=False)
ap.add_argument("-plottraining",
                required=False,
                type=bool,
                help='Plot loss and accuracy during training. File will be added to path.',
                default=True)
ap.add_argument("-showconv",
                required=False,
                type=bool,
                help='Plot internal representation of the convolution layers',
                default=False)
ap.add_argument("-savenetwork",
                required=False,
                type=bool,
                help='Save trained network to path',
                default=False)
args = vars(ap.parse_args())


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
    # Get correct network
    if args['network'] == 'Net':
        network = nets.Net()
    elif args['network'] == 'JNet1':
        network = nets.JNet1()
    elif args['network'] == 'JNet2':
        network = nets.JNet2()
    elif args['network'] == 'JNet3':
        network = nets.JNet3()
    elif args['network'] == 'JNet4':
        network = nets.JNet4()
    elif args['network'] == 'JNet5':
        network = nets.JNet5()
    elif args['network'] == 'JNet6':
        network = nets.JNet6()
    elif args['network'] == 'JNet7':
        network = nets.JNet7()
    elif args['network'] == 'JNet8':
        network = nets.JNet8()
    elif args['network'] == 'JNet9':
        network = nets.JNet9()
    elif args['network'] == 'JNet10':
        network = nets.JNet10()
    elif args['network'] == 'JNet11':
        network = nets.JNet11()
    elif args['network'] == 'JNet12':
        network = nets.JNet12()
    elif args['network'] == 'JNet13':
        network = nets.JNet13()
    elif args['network'] == 'JNet14':
        network = nets.JNet14()
    elif args['network'] == 'JNet15':
        network = nets.JNet15()
    elif args['network'] == 'JNet16':
        network = nets.JNet16()
    elif args['network'] == 'JNet17':
        network = nets.JNet17()
    elif args['network'] == 'JNet18':
        network = nets.JNet18()

    # Set parameters
    plotData, plotResult, plotTraining, showConvolutionLayer = args['plotdata'], args['plotresult'], args['plottraining'], args['showconv']
    saveNetwork = args['savenetwork']
    batchsize = args['bs']
    interval = args['i']
    n_epochs = args['ne']
    learning_rate = args['lr']
    decay = args['decay']
    loader_type = args['lt']

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
