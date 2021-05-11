# Imports
import torch
import torchvision
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, TensorDataset, random_split

df = pd.read_csv('data/FER/fer2013.csv')

df_train = df[df['Usage']=='Training']
df_valid = df[df['Usage']=='PrivateTest']
df_test = df[df['Usage']=='PublicTest']

print(df.head())
