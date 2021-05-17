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


# trainloader, testloader, _, _ = loaders.make_loader_small(1)
network_path = 'networks/JNet15_nepoch50_lr0.001_batchsize25_loaderbalanced.pth'
#
# visualiser.show_convolution_layers(network_path, trainloader)
#
# visualiser.visualise_important_area(network_path, trainloader)
#
# visualiser.visualise_network(network_path, trainloader)

_, testloader, _, _ = loaders.make_loader(16)
network = torch.load(network_path, map_location=torch.device('cpu'))
visualiser.testshow(testloader,network,4,4)

determine_performance(network, testloader)
