import numpy as np
import os
from torch import nn
import torch
import loaders
import nets
import visualiser
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt

trainloader, _, _, _ = loaders.make_loader_small(40)
visualiser.show_convolution_layers('ConvNet_nepoch25_lr0.001_batchsize25.pth', trainloader)
