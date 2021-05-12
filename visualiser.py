import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def trainshow(loader, n, m, classes):
    # Get data
    dataiter = iter(loader)
    data, label = dataiter.next()

    # Adjust parameters
    n = n if n*m < data.shape[0] else int(data.shape[0]**.5)
    m = m if n*m < data.shape[0] else int(data.shape[0]**.5)

    # Make subplot
    fig, ax = plt.subplots(n,m)

    # Plot result
    for i in range(n):
        for j in range(m):
            image = data[i*m+j][0]
            ax[i,j].imshow(image, cmap='gray')
            ax[i,j].axis('off')
            ax[i,j].title.set_text(classes[int(label[i*m+j])])
    plt.show()

def testshow(loader, network, n, m, classes):
    # Get data
    dataiter = iter(loader)
    data, label = dataiter.next()

    # Adjust parameters
    n = n if n*m < data.shape[0] else int(data.shape[0]**.5)
    m = m if n*m < data.shape[0] else int(data.shape[0]**.5)

    # Get prediction
    output = network(data)
    pred = output.data.max(1, keepdim=True)[1]

    # Adjust parameters
    n = n if n*m < data.shape[0] else int(data.shape[0]**.5)
    m = m if n*m < data.shape[0] else int(data.shape[0]**.5)

    # Make subplot
    fig, ax = plt.subplots(n,m)

    # Plot result
    for i in range(n):
        for j in range(m):
            image = data[i*m+j][0]
            ax[i,j].imshow(image, cmap='gray')
            ax[i, j].title.set_text("Predicted: {}.\nTrue: {}.".format(classes[int(pred[i*m+j])],
                                                                      classes[int(label[i*m+j])]))
            ax[i, j].axis('off')
    plt.show()


def plot_training(train_counter, train_losses, train_accuracy, valid_counter, valid_losses, valid_accuracy):
    fig, ax = plt.subplots(1, 2)

    # Loss function
    print(len(train_counter),len(train_losses), len(valid_counter), len(valid_losses))
    ax[0].plot(train_counter, train_losses, color='cornflowerblue')
    ax[0].scatter(valid_counter, valid_losses, color='red')
    ax[0].legend(['Train Loss', 'Validation Loss'], loc='upper right')
    ax[0].set_xlabel('number of training examples seen')
    ax[0].set_ylabel('cross entropy loss')
    ax[0].title.set_text('Loss')

    # Accuracy
    ax[1].plot(valid_counter[1:], train_accuracy, color='cornflowerblue')
    ax[1].plot(valid_counter, valid_accuracy, color='orange')
    ax[1].legend(['Train Accuracy', 'Validation Accuracy'], loc='lower right')
    ax[1].set_xlabel('number of training examples seen')
    ax[1].set_ylabel('Accuracy')
    ax[1].title.set_text('Accuracy')

    plt.show()
