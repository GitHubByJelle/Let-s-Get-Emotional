# import datasets in torchvision
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchfunc

import loaders
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# import model zoo in torchvision

if __name__ == '__main__':
    network = torch.load("ConvNet_nepoch25_lr0.001_batchsize4_loaderbalancedtransform.pth")

    trainloader, valloader, testloader, plotloader = loaders.make_loader_small(10)

    # Recorder saving inputs to all submodules
    recorder = torchfunc.hooks.recorders.ForwardPre()

    # Will register hook for all submodules of resnet18
    # You could specify some submodules by index or by layer type, see docs
    recorder.modules(network)

    # Push example image through network
    network(torch.randn(1, 1, 48, 48))

    # Zero image before going into the third submodule of this network
    print(recorder.data[3][0].shape)

    # # You can see all submodules and their positions by running this:
    # for i, submodule in enumerate(network.modules()):
    #     print(i, submodule)
    #
    # # Or you can just print the network to get this info
    # You can see all submodules and their positions by running this:
    next = False
    for i, submodule in enumerate(network.modules()):
        if next:
            data = recorder.data[i][0]
            width, height = int(np.ceil(data.shape[1]**.5)), int(np.ceil(data.shape[1]**.5))
            print(width)

            fig, ax = plt.subplots(width, height)
            for r in range(height):
                for c in range(width):
                    if r * width + c < data.shape[1]:
                        ax[r,c].imshow(data[0,r*width+c,:,:].detach().numpy(), cmap='gray')

                    ax[r,c].axis('off')

            plt.show()

        next = type(submodule) == nn.Conv2d
