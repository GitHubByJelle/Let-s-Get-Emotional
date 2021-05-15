import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn
from torchviz import make_dot
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

classes = {0:'anger',1:'disgust',2:'fear',3:'happiness',4:'sadness',5:'surprise',6:'neutral'}

def trainshow(loader, n, m):
    # Get data
    dataiter = iter(loader)
    data, label = dataiter.next()

    # Adjust parameters
    n = n if n * m < data.shape[0] else int(data.shape[0] ** .5)
    m = m if n * m < data.shape[0] else int(data.shape[0] ** .5)

    # Make subplot
    fig, ax = plt.subplots(n, m)

    # Plot result
    for i in range(n):
        for j in range(m):
            image = data[i * m + j][0]
            ax[i, j].imshow(image, cmap='gray')
            ax[i, j].axis('off')
            ax[i, j].title.set_text(classes[int(label[i * m + j])])
    plt.show()


def testshow(loader, network, n, m):
    # Get data
    dataiter = iter(loader)
    data, label = dataiter.next()

    # Adjust parameters
    n = n if n * m < data.shape[0] else int(data.shape[0] ** .5)
    m = m if n * m < data.shape[0] else int(data.shape[0] ** .5)

    # Get prediction
    output = network(data)
    pred = output.data.max(1, keepdim=True)[1]

    # Adjust parameters
    n = n if n * m < data.shape[0] else int(data.shape[0] ** .5)
    m = m if n * m < data.shape[0] else int(data.shape[0] ** .5)

    # Make subplot
    fig, ax = plt.subplots(n, m)

    # Plot result
    for i in range(n):
        for j in range(m):
            image = data[i * m + j][0]
            ax[i, j].imshow(image, cmap='gray')
            ax[i, j].title.set_text("Predicted: {}.\nTrue: {}.".format(classes[int(pred[i * m + j])],
                                                                       classes[int(label[i * m + j])]))
            ax[i, j].axis('off')
    plt.show()


def plot_training(train_counter, train_losses, train_accuracy, valid_counter, valid_losses, valid_accuracy, title,
                  fn):
    fig, ax = plt.subplots(1, 2, figsize=(15,7))

    # Loss function
    ax[0].plot(train_counter, train_losses, color='cornflowerblue', zorder=0)
    ax[0].scatter(valid_counter, valid_losses, color='red', zorder=1)
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

    fig.suptitle(title)

    plt.savefig(fn)
    # plt.show()


def show_convolution_layers(model_path, loader):
    ### Implementation from Sovit Ranjan Rath.
    ### https://debuggercafe.com/visualizing-filters-and-feature-maps-in-convolutional-neural-networks-using-pytorch/
    ### Code has been adjusted to fit own code
    model = torch.load(model_path)
    model.eval()

    # initialise weights and layers
    model_weights = []  # we will save the conv layer weights in this list
    conv_layers = []  # we will save the 49 conv layers in this list
    # get all the model children as list
    model_children1 = list(model.children())

    # Determine all convolution layers
    model_children = []
    for i, child in enumerate(model_children1):
        if type(child) == nn.Sequential:
            for c in child:
                if type(c) == nn.Conv2d:
                    model_children.append(c)
        elif type(child) == nn.Conv2d:
            model_children.append(child)

    # counter to keep count of the conv layers
    counter = 0
    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolutional layers: {counter}")

    # take a look at the conv layers and the respective weights
    for weight, conv in zip(model_weights, conv_layers):
        # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
        print(f"CONV: {conv} ====> SHAPE: {weight.shape}")

    # visualize the first conv layer filters
    plt.figure(figsize=(20, 17))
    for layer in range(len(conv_layers)):
        for i, filter in enumerate(model_weights[layer]):
            plt.subplot(np.ceil(len(model_weights[layer]) ** .5), np.ceil(len(model_weights[layer]) ** .5),
                        i + 1)  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
            plt.imshow(filter[0, :, :].detach(), cmap='gray')
            plt.axis('off')
            plt.suptitle('Feature maps convolution layer {}'.format(layer))
        plt.show()

    # Get data
    dataiter = iter(loader)
    data, label = dataiter.next()

    # Get images
    img = data[0:3, :, :, :]

    # pass the image through all the layers
    results = [conv_layers[0](img)]
    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        results.append(conv_layers[i](results[-1]))
    # make a copy of the `results`
    outputs = results

    # visualize features
    # (although there are more feature maps in the upper layers)
    for num_layer in range(len(outputs)):
        plt.figure(figsize=(30, 30))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        for i, filter in enumerate(layer_viz):
            if i == 64:  # limit on 64
                break
            plt.subplot(np.ceil(layer_viz.size()[0] ** .5), np.ceil(layer_viz.size()[0] ** .5), i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
        plt.suptitle("Output of feature maps convolution layer {}".format(num_layer))
        # print(f"Saving layer {num_layer} feature maps...")
        # plt.savefig(f"../outputs/layer_{num_layer}.png")
        # plt.show()
        # plt.close()
        plt.show()

def visualise_network(path, loader):
    """
    Be sure that GraphViz is installed on your OP (and added to your path). And be sure that the package is installed.
    :param path: path to network file after training
    :param loader: dataloader
    :return: png image of network (gets saved)
    """
    # Get data
    dataiter = iter(loader)
    data, label = dataiter.next()

    # Get output
    network = torch.load(path)
    out = network(data)

    # Make visualisation
    make_dot(out, params=dict(network.named_parameters()), show_attrs=True).render(path.rstrip('.pth'), format="png")

def occlusion(model, image, label, occ_size=50, occ_stride=50, occ_pixel=0.5):
    #### Implementation from Niranjan Kumar
    #### https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e
    # get the width and height of the image
    width, height = image.shape[-2], image.shape[-1]

    # setting the output image width and height
    output_height = int(np.ceil((height - occ_size) / occ_stride))
    output_width = int(np.ceil((width - occ_size) / occ_stride))

    # create a white image of sizes we defined
    heatmap = torch.zeros((output_height, output_width))

    # iterate all the pixels in each column
    for h in range(0, height):
        for w in range(0, width):

            h_start = h * occ_stride
            w_start = w * occ_stride
            h_end = min(height, h_start + occ_size)
            w_end = min(width, w_start + occ_size)

            if (w_end) >= width or (h_end) >= height:
                continue

            input_image = image.clone().detach()

            # replacing all the pixel information in the image with occ_pixel(grey) in the specified location
            input_image[:, :, w_start:w_end, h_start:h_end] = occ_pixel

            # run inference on modified image
            output = model(input_image)
            output = nn.functional.softmax(output, dim=1)
            prob = output.tolist()[0][label]

            # setting the heatmap location to probability value
            heatmap[h, w] = prob

    return heatmap

def visualise_important_area(network_path, loader):
    # Get network
    network = torch.load(network_path)

    # Get image and label
    dataiter = iter(loader)
    data, label = dataiter.next()
    image = data[0][0]
    label = int(label[0])

    # Get prediction on image
    pred = int(network(data).data.max(1, keepdim=False)[1][0])

    # Make heatmap by using occlusion
    heatmap = occlusion(network, data, label, occ_size=7, occ_stride=3, occ_pixel=.5)

    # Make subplots
    fig, ax = plt.subplots(1,3)

    # Show original image
    ax[0].imshow(image, cmap='gray')

    # Black part -> lower probability -> occlusion is really effective, so part is really important
    ax[1].imshow(heatmap, cmap='gray')

    # Image and occulusion
    heatmap = cv2.resize(np.float32(heatmap), image.shape, interpolation = cv2.INTER_CUBIC)
    ax[2].imshow(image, cmap='gray')
    ax[2].imshow(1-heatmap, cmap='viridis',alpha=.4)

    # Cosmetics
    # Remove axis
    for i in range(3):
        ax[i].axis('off')
    # Main title
    plt.suptitle("Important areas of image")

    # Subplot titles
    ax[0].title.set_text("Orginal image\n True: '{}', Predicted: '{}'".format(classes[label], classes[pred]))
    ax[1].title.set_text("Occlusion map")
    ax[2].title.set_text("Important areas on image")

    # Show image
    plt.show()
