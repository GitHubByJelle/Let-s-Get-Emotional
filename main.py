import torch
import os
import torch.nn.functional as F
import torch.optim as optim
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
            # torch.save(network.state_dict(), '/results/model.pth')
            # torch.save(optimizer.state_dict(), '/results/optimizer.pth')
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
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(loader.dataset),
            accuracy))


if __name__ == '__main__':
    ###### VARIABLES TO CHANGE
    plotData, plotResult, plotTraining = True, True, True
    batchsize = 25
    interval = 5
    n_epochs = 5
    learning_rate = 0.001
    network = nets.ExampleNet3()

    # Define loader (normal, balanced or transform)
    trainloader, valloader, testloader, plotloader = loaders.make_balanced_loader(batchsize)

    # Define optimizer
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    ###### START NETWORK
    # Show network
    print(network)

    # Define classes
    classes = {0:'anger',1:'disgust',2:'fear',3:'happiness',4:'sadness',5:'surprise',6:'neutral'}

    # Plot train data
    if plotData:
        visualiser.trainshow(trainloader, 10, 10, classes)

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

    # Visualise test set
    if plotResult:
        visualiser.testshow(testloader,network,4,4,classes)

    # Plot training progress
    if plotTraining:
        visualiser.plot_training(train_counter, train_losses, train_accuracy,
                                 valid_counter, valid_losses, valid_accuracy)
