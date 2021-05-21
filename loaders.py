import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, TensorDataset
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def make_loader(batch_size):
    """
    Creates a loader for the data
    :param batch_size: size of the batches
    :return: loader for train, test, validate and plot
    """
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('data/FER/fer2013.csv')

    # Make TensorDataset Training
    print("Creating training set...")
    df_train = df[df['Usage']=='Training']
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
    trainloader = DataLoader(trainset, batch_size)
    valloader = DataLoader(validset, batch_size)
    testloader = DataLoader(testset, batch_size)
    plotloader = DataLoader(testset, 1, shuffle=True)

    # Return loaders
    return trainloader, valloader, testloader, plotloader

def make_loader_small(batch_size, size=100):
    """
    Creates a smaller loader for the data (random samples data)
    Can be used for debugging.
    :param batch_size: size of the batches
    :return: loader for train, test, validate and plot
    """
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('data/FER/fer2013.csv')
    df = df.sample(size)

    # Make TensorDataset Training
    print("Creating training set...")
    df_train = df[df['Usage']=='Training']
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
    trainloader = DataLoader(trainset, batch_size)
    valloader = DataLoader(validset, batch_size)
    testloader = DataLoader(testset, batch_size)
    plotloader = DataLoader(testset, 1, shuffle=True)

    # Return loaders
    return trainloader, valloader, testloader, plotloader

def make_balanced_loader(batch_size, printBalance=False):
    """
    Creates a oversampled loader for the data
    :param batch_size: size of the batches
    :param printBalance: Show the proportion of every batch
    :return: loader for train, test, validate and plot
    """
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('data/FER/fer2013.csv')
    # df = df.sample(1000)

    # Make TensorDataset Training
    print("Creating training set...")
    df_train = df[df['Usage']=='Training']

    # Calculate balance weights
    class_sample_count = np.array(
        [len(np.where(df_train.emotion == t)[0]) for t in np.unique(df_train.emotion)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in df_train.emotion])

    # Create sampler
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # Create trainset
    train_input = df_train.assign(pixels = df_train.pixels.str.split(" ")) \
        .pixels.apply(pd.Series) \
        .values.reshape(-1,1,48,48).astype('float32') / 255
    train_output = df_train.emotion.values
    trainset = TensorDataset(torch.from_numpy(train_input.astype('float32')),torch.from_numpy(train_output.astype('float32')))

    # Make TensorDataset Valid - No balance for validation set
    print("Creating validation set...")
    df_valid = df[df['Usage']=='PrivateTest']
    valid_input = df_valid.assign(pixels = df_valid.pixels.str.split(" ")) \
        .pixels.apply(pd.Series) \
        .values.reshape(-1,1,48,48).astype('float32') / 255
    valid_output = df_valid.emotion.values
    validset = TensorDataset(torch.from_numpy(valid_input.astype('float32')),torch.from_numpy(valid_output.astype('float32')))

    # Make TensorDataset Test - No balance for test set
    print("Creating test set...")
    df_test = df[df['Usage']=='PublicTest']
    test_input = df_test.assign(pixels = df_test.pixels.str.split(" ")) \
        .pixels.apply(pd.Series) \
        .values.reshape(-1,1,48,48).astype('float32') / 255
    test_output = df_test.emotion.values
    testset = TensorDataset(torch.from_numpy(test_input.astype('float32')),torch.from_numpy(test_output.astype('float32')))

    # Create loaders
    print("Creating loaders")
    trainloader = DataLoader(trainset, batch_size, sampler=sampler)
    valloader = DataLoader(validset, batch_size)
    testloader = DataLoader(testset, batch_size)
    plotloader = DataLoader(testset, 1, shuffle=True)

    if printBalance:
        for t in np.unique(df_train.emotion):
            print('target train {}: {}'.format(t,len(np.where(df_train.emotion==t)[0])))

        for i, (data, target) in enumerate(trainloader):
            print("batch index {}, 0/1/2/3/4/5/6: {}/{}/{}/{}/{}/{}/{}".format(
                i,
                len(np.where(target.numpy() == 0)[0]),
                len(np.where(target.numpy() == 1)[0]),
                len(np.where(target.numpy() == 2)[0]),
                len(np.where(target.numpy() == 3)[0]),
                len(np.where(target.numpy() == 4)[0]),
                len(np.where(target.numpy() == 5)[0]),
                len(np.where(target.numpy() == 6)[0])))

    # Return loaders
    return trainloader, valloader, testloader, plotloader

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

def make_transform_loader(batch_size):
    """
    Creates a loader for the data that augments data
    :param batch_size: size of the batches
    :return: loader for train, test, validate and plot
    """
    print("Loading dataset...")
    with open('data/FER/fer2013.csv', 'r') as f:
        data = f.readlines()

    data = data[1:]
    # data = [data[i] for i in np.random.randint(0,len(data),1000)]

    # Images
    print("Extracting images...")
    X = np.array([[j.split(' ') for j in i.strip().split(',')][1] for i in data]).astype('float32') / 255
    X = X.reshape((-1, 1, 48, 48))
    X = torch.tensor(X)

    # Labels
    print("Extracting labels")
    y = np.array([int(i[0]) for i in data])
    # y = y.reshape(y.shape[0],1)
    y = torch.tensor(y).type(torch.LongTensor)

    # Usage
    print("Extracting usage...")
    usage = np.array([i.strip().split(',')[2] for i in data])

    del data

    # Create trainloader
    print("Creating train loader....")
    x_train, y_train = X[usage=='Training'], y[usage=='Training']
    transform = transforms.Compose([
        transforms.RandomCrop(42),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize((x_train.mean()), (x_train.std()))
    ])
    trainset = CustomTensorDataset(tensors=(x_train, y_train), transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Create testloader
    print("Creating test loader....")
    x_test, y_test = X[usage=='PublicTest'], y[usage=='PublicTest']
    testset = CustomTensorDataset(tensors=(x_test, y_test), transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

    # Create valloader
    print("Creating validation loader....")
    x_val, y_val = X[usage=='PrivateTest'], y[usage=='PrivateTest']
    valset = CustomTensorDataset(tensors=(x_val, y_val), transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size)

    # Create plotloader
    print("Creating plot loader....")
    plotloader = torch.utils.data.DataLoader(testset, batch_size=2)

    return trainloader, valloader, testloader, plotloader

def make_balanced_transform_loader(batch_size, printBalance=False):
    """
    Creates a loader for the data that augments and oversamples the data
    :param batch_size: size of the batches
    :param printBalance: Show the proportion of every batch
    :return: loader for train, test, validate and plot
    """
    print("Loading dataset...")
    with open('data/FER/fer2013.csv', 'r') as f:
        data = f.readlines()

    data = data[1:]
    # data = [data[i] for i in np.random.randint(0,len(data),1000)]

    # Images
    print("Extracting images...")
    X = np.array([[j.split(' ') for j in i.strip().split(',')][1] for i in data]).astype('float32') / 255
    X = X.reshape((-1, 1, 48, 48))
    X = torch.tensor(X)

    # Labels
    print("Extracting labels")
    y = np.array([int(i[0]) for i in data])
    # y = y.reshape(y.shape[0],1)
    y = torch.tensor(y).type(torch.LongTensor)

    # Usage
    print("Extracting usage...")
    usage = np.array([i.strip().split(',')[2] for i in data])

    del data

    # Create trainloader
    print("Creating train loader....")
    x_train, y_train = X[usage=='Training'], y[usage=='Training']

    # Calculate balance weights
    class_sample_count = np.array(
        [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])

    # Create sampler
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    transform = transforms.Compose([
        # transforms.RandomCrop(42),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize((x_train.mean()), (x_train.std()))
    ])
    trainset = CustomTensorDataset(tensors=(x_train, y_train), transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=sampler)

    # Create testloader
    print("Creating test loader....")
    x_test, y_test = X[usage=='PublicTest'], y[usage=='PublicTest']
    testset = CustomTensorDataset(tensors=(x_test, y_test), transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

    # Create valloader
    print("Creating validation loader....")
    x_val, y_val = X[usage=='PrivateTest'], y[usage=='PrivateTest']
    valset = CustomTensorDataset(tensors=(x_val, y_val), transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size)

    # Create plotloader
    print("Creating plot loader....")
    plotloader = torch.utils.data.DataLoader(testset, batch_size=2)

    if printBalance:
        for t in np.unique(y_train):
            print('target train {}: {}'.format(t,len(np.where(y_train==t)[0])))

        for i, (data, target) in enumerate(trainloader):
            print("batch index {}, 0/1/2/3/4/5/6: {}/{}/{}/{}/{}/{}/{}".format(
                i,
                len(np.where(target.numpy() == 0)[0]),
                len(np.where(target.numpy() == 1)[0]),
                len(np.where(target.numpy() == 2)[0]),
                len(np.where(target.numpy() == 3)[0]),
                len(np.where(target.numpy() == 4)[0]),
                len(np.where(target.numpy() == 5)[0]),
                len(np.where(target.numpy() == 6)[0])))

    return trainloader, valloader, testloader, plotloader
