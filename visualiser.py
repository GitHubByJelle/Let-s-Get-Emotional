import matplotlib.pyplot as plt
import numpy as np
import torch


def trainshow(loader, n, m, classes):
    fig, ax = plt.subplots(n, m)
    for i in range(n):
        for j in range(m):
            dataiter = iter(loader)
            image, label = dataiter.next()
            ax[i, j].imshow(np.squeeze(image), cmap='gray')
            ax[i, j].title.set_text(classes[label.numpy()[0]])
            ax[i, j].axis('off')

    plt.show()


def testshow(loader, network, n, m, classes):
    fig, ax = plt.subplots(n, m)
    for i in range(n):
        for j in range(m):
            dataiter = iter(loader)
            image, label = dataiter.next()
            output = network(image)
            pred = torch.max(output, dim=1).indices.numpy()[0]
            true = label.numpy()[0]
            ax[i, j].imshow(np.squeeze(image), cmap='gray')
            ax[i, j].title.set_text("Predicted: {}. True: {}.".format(classes[pred], classes[true]))
            ax[i, j].axis('off')

    plt.show()


def plot_training(train_counter, train_losses, train_accuracy, test_counter, test_losses, test_accuracy):
    fig, ax = plt.subplots(1, 2)

    # Loss function
    ax[0].plot(train_counter, train_losses, color='cornflowerblue')
    ax[0].scatter(test_counter, test_losses, color='red')
    ax[0].legend(['Train Loss', 'Test Loss'], loc='upper right')
    ax[0].set_xlabel('number of training examples seen')
    ax[0].set_ylabel('cross entropy loss')
    ax[0].title.set_text('Loss')

    # Accuracy
    ax[1].plot(test_counter[1:], train_accuracy, color='cornflowerblue')
    ax[1].plot(test_counter, test_accuracy, color='orange')
    ax[1].legend(['Train Accuracy', 'Test Accuracy'], loc='upper right')
    ax[1].set_xlabel('number of training examples seen')
    ax[1].set_ylabel('Accuracy')
    ax[1].title.set_text('Accuracy')

    plt.show()
