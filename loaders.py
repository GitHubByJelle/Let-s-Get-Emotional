import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def make_loader(batch_size):
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('data/FER/fer2013.csv')
    df = df.sample(1000)

    # Make TensorDataset Training
    print("Creating training set...")
    df_train = df[df['Usage']=='Training']

    class_sample_count = np.array(
        [len(np.where(df_train.emotion == t)[0]) for t in np.unique(df_train.emotion)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in df_train.emotion])

    samples_weight = torch.from_numpy(samples_weight)
    # samples_weigth = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_input = df_train.assign(pixels = df_train.pixels.str.split(" ")) \
        .pixels.apply(pd.Series) \
        .values.reshape(-1,1,48,48).astype('float32') / 255
    train_output = df_train.emotion.values
    trainset = TensorDataset(torch.from_numpy(train_input.astype('float32')),torch.from_numpy(train_output.astype('float32')))

    # Make TensorDataset Valid
    print("Creating validation set...")
    df_valid = df[df['Usage']=='PrivateTest']
    valid_input = df_valid.assign(pixels = df_valid.pixels.str.split(" ")) \
        .pixels.apply(pd.Series) \
        .values.reshape(-1,1,48,48).astype('float32') / 255
    valid_output = df_valid.emotion.values
    validset = TensorDataset(torch.from_numpy(valid_input.astype('float32')),torch.from_numpy(valid_output.astype('float32')))

    # Make TensorDataset Test
    print("Creating test set...")
    df_test = df[df['Usage']=='PublicTest']
    test_input = df_test.assign(pixels = df_test.pixels.str.split(" ")) \
        .pixels.apply(pd.Series) \
        .values.reshape(-1,1,48,48).astype('float32') / 255
    test_output = df_test.emotion.values
    testset = TensorDataset(torch.from_numpy(test_input.astype('float32')),torch.from_numpy(test_output.astype('float32')))

    # Create loaders
    print("Creating loaders")
    # for t in np.unique(df_train.emotion):
    #     print('target train {}: {}'.format(t,len(np.where(df_train.emotion==t)[0])))
    trainloader = DataLoader(trainset, batch_size, sampler=sampler)
    valloader = DataLoader(validset, batch_size)
    testloader = DataLoader(testset, batch_size)
    plotloader = DataLoader(testset, 1, shuffle=True)

    # for i, (data, target) in enumerate(trainloader):
    #     print("batch index {}, 0/1/2/3: {}/{}/{}/{}".format(
    #         i,
    #         len(np.where(target.numpy() == 0)[0]),
    #         len(np.where(target.numpy() == 1)[0]),
    #         len(np.where(target.numpy() == 2)[0]),
    #         len(np.where(target.numpy() == 3)[0]),
    #         len(np.where(target.numpy() == 4)[0]),
    #         len(np.where(target.numpy() == 5)[0]),
    #         len(np.where(target.numpy() == 6)[0])))

    # Return loaders
    return trainloader, valloader, testloader, plotloader
