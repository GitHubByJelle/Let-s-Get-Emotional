import numpy as np
import os
from torch import nn
import torch
import loaders
import nets
import visualiser
import matplotlib.pyplot as plt
from torchviz import make_dot
from main import determine_performance
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

# # Load part of dataset and define path
# trainloader, testloader, _, _ = loaders.make_loader_small(1)
# network_path = 'networks/JNet15_nepoch50_lr0.001_batchsize25_loaderbalanced.pth'
#
# # Visualise convolution layers
# visualiser.show_convolution_layers(network_path, trainloader)
#
# # Visualise the occlusion map
# visualiser.visualise_important_area(network_path, trainloader)
#
# # Visualise and save network
# visualiser.visualise_network(network_path, trainloader)
#
# # Load all data and network
# _, testloader, _, _ = loaders.make_loader(16)
# network = torch.load(network_path, map_location=torch.device('cpu'))
#
# # Visualise predicting examples (true and predicted label)
# visualiser.testshow(testloader,network,4,4)
#
# Determine performance
# determine_performance(network, testloader)

#### For different networks, print performance
# Get loader
_, testloader, _, _ = loaders.make_loader(16)

# Select models
network_numbers = [1, 5, 8, 2, 9, 7, 13, 15, 17, 18]
for num in network_numbers:
    # Load correct network
    network_path = 'networks/JNet{}_nepoch50_lr0.001_batchsize25_loaderbalanced.pth'.format(num)
    network = torch.load(network_path, map_location=torch.device('cpu'))

    # Print performance
    print("Model {}:".format(num))
    determine_performance(network, testloader)
