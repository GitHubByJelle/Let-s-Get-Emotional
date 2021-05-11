from torch.utils.data import WeightedRandomSampler, DataLoader
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

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


def imshow(img, title=''):
    """Plot the image batch.
    """
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(np.transpose( img.numpy(), (1, 2, 0)), cmap='gray')
    plt.show()

# Let's add some transforms

# Dataset with flipping tranformations

def vflip(tensor):
    """Flips tensor vertically.
    """
    tensor = tensor.flip(1)
    return tensor


def hflip(tensor):
    """Flips tensor horizontally.
    """
    tensor = tensor.flip(2)
    return tensor

if __name__ == '__main__':
    # numDataPoints = 1000
    # data_dim = 5
    # bs = 100
    #
    # # Create dummy data with class imbalance 9 to 1
    # data = torch.FloatTensor(numDataPoints, data_dim)
    # target = np.hstack((np.zeros(int(numDataPoints * 0.8), dtype=np.int32),
    #                     np.ones(int(numDataPoints * 0.1), dtype=np.int32),
    #                     np.ones(int(numDataPoints * 0.07), dtype=np.int32)*2,
    #                     np.ones(int(numDataPoints * 0.03), dtype=np.int32)*3))
    #
    # for t in np.unique(target):
    #     print('target train {}: {}'.format(t,len(np.where(target==t)[0])))
    #
    # class_sample_count = np.array(
    #     [len(np.where(target == t)[0]) for t in np.unique(target)])
    # weight = 1. / class_sample_count
    # samples_weight = np.array([weight[t] for t in target])
    #
    # samples_weight = torch.from_numpy(samples_weight)
    # # samples_weigth = samples_weight.double()
    # sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    #
    # target = torch.from_numpy(target).long()
    # train_dataset = torch.utils.data.TensorDataset(data, target)
    #
    # train_loader = DataLoader(
    #     train_dataset, batch_size=bs, num_workers=1, sampler=sampler)
    #
    # for i, (data, target) in enumerate(train_loader):
    #     print("batch index {}, 0/1/2/3: {}/{}/{}/{}".format(
    #         i,
    #         len(np.where(target.numpy() == 0)[0]),
    #         len(np.where(target.numpy() == 1)[0]),
    #         len(np.where(target.numpy() == 2)[0]),
    #         len(np.where(target.numpy() == 3)[0])))

    # Import mnist dataset from cvs file and convert it to torch tensor

    with open('data/FER/fer2013.csv', 'r') as f:
        mnist_train = f.readlines()

    mnist_train = mnist_train[1:10]

    # Images
    X_train = np.array([[j.split(' ') for j in i.strip().split(',')][1] for i in mnist_train]).astype('float32') / 255
    X_train = X_train.reshape((-1, 1, 48, 48))
    X_train = torch.tensor(X_train)

    # Labels
    y_train = np.array([int(i[0]) for i in mnist_train])
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_train = torch.tensor(y_train).type(torch.LongTensor)

    del mnist_train

    # Dataset w/o any tranformations
    train_dataset_normal = CustomTensorDataset(tensors=(X_train, y_train), transform=None)
    train_loader = torch.utils.data.DataLoader(train_dataset_normal, batch_size=16)

    # iterate
    for i, data in enumerate(train_loader):
        x, y = data
        print(x, x.shape)
        imshow(torchvision.utils.make_grid(x, 4), title='Normal')
        break  # we need just one batch


    train_dataset_vf = CustomTensorDataset(tensors=(X_train, y_train), transform=vflip)
    train_loader = torch.utils.data.DataLoader(train_dataset_vf, batch_size=16)

    result = []

    for i, data in enumerate(train_loader):
        x, y = data
        imshow(torchvision.utils.make_grid(x, 4), title='Vertical flip')
        break


    train_dataset_hf = CustomTensorDataset(tensors=(X_train, y_train), transform=hflip)
    train_loader = torch.utils.data.DataLoader(train_dataset_hf, batch_size=16)

    result = []

    for i, data in enumerate(train_loader):
        x, y = data
        imshow(torchvision.utils.make_grid(x, 4), title='Horizontal flip')
        break

    train_dataset_transform = CustomTensorDataset(tensors=(X_train, y_train), transform=transforms.Compose([
        transforms.RandomCrop(42),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Normalize((X_train.mean()), (X_train.std()))]))
    train_loader = torch.utils.data.DataLoader(train_dataset_transform, batch_size=16)

    print(X_train.shape,(X_train.mean()), (X_train.std()))

    result = []

    for i, data in enumerate(train_loader):
        x, y = data
        imshow(torchvision.utils.make_grid(x, 4), title='Transform')
        break

