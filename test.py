import numpy as np
import os
from torch import nn
import torch
import loaders
import nets
import visualiser
import matplotlib.pyplot as plt
from torchviz import make_dot
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'


trainloader, _, _, _ = loaders.make_loader_small(40)

visualiser.show_convolution_layers('ConvNet_nepoch25_lr0.001_batchsize25.pth', trainloader)

visualiser.visualise_network('ConvNet_nepoch25_lr0.001_batchsize25.pth', trainloader)

