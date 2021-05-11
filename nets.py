import torch.nn as nn
import torch.nn.functional as F

class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 9 * 9, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=7)

    def forward(self, x):
        # (1) input layer
        t = x

        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) hidden linear layer
        t = t.reshape(-1, 12 * 9 * 9)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output layer
        t = self.out(t)
        t = F.softmax(t, dim=1)

        return t

class ExampleNet2(nn.Module):
    def __init__(self):
        super(ExampleNet2, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5)
        self.batchnorm_1 = nn.BatchNorm2d(64)
        self.conv2d_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.batchnorm_2 = nn.BatchNorm2d(64)
        self.dropout_1 = nn.Dropout2d(p=.4)

        self.conv2d_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.batchnorm_3 = nn.BatchNorm2d(128)
        self.conv2d_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.batchnorm_4 = nn.BatchNorm2d(128)
        self.dropout_2 = nn.Dropout2d(p=.4)

        self.conv2d_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.batchnorm_5 = nn.BatchNorm2d(256)
        self.conv2d_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.batchnorm_6 = nn.BatchNorm2d(256)
        self.dropout_3 = nn.Dropout2d()

        self.linear1 = nn.Linear(in_features=256*2*2,out_features=128)
        self.batchnorm_7 = nn.BatchNorm1d(128)
        self.dropout_4 = nn.Dropout2d(p=.6)

        self.linear2 = nn.Linear(in_features=128, out_features=7)

    def forward(self, x):
        # (1) input layer
        x = x

        # (2) hidden conv layer (block 1)
        x = self.conv2d_1(x)
        x = F.elu(x)
        x = self.batchnorm_1(x)

        # (3) hidden conv layer (block 1)
        x = self.conv2d_2(x)
        x = F.elu(x)
        x = self.batchnorm_2(x)

        # (4) end block 1
        x = F.max_pool2d(x, kernel_size=2)
        x = self.dropout_1(x)

        # (5) hidden conv layer (block 2)
        x = self.conv2d_3(x)
        x = F.elu(x)
        x = self.batchnorm_3(x)

        # (6) hidden conv layer (block 2)
        x = self.conv2d_4(x)
        x = F.elu(x)
        x = self.batchnorm_4(x)

        # (7) end block 2
        x = F.max_pool2d(x, kernel_size=2)
        x = self.dropout_2(x)

        # (8) hidden conv layer (block 3)
        x = self.conv2d_5(x)
        x = F.elu(x)
        x = self.batchnorm_5(x)

        # (9) hidden conv layer (block 3)
        x = self.conv2d_6(x)
        x = F.elu(x)
        x = self.batchnorm_6(x)

        # (10) end block 3
        x = F.max_pool2d(x, kernel_size=2)
        x = self.dropout_3(x)

        # (11) Linear layer
        x = x.reshape(-1, 256*2*2)
        x = self.linear1(x)
        x = self.batchnorm_7(x)
        x = self.dropout_4(x)

        # Output layer
        x = self.linear2(x)
        x = F.softmax(x, dim=1)

        return x

class ExampleNet3(nn.Module):
    def __init__(self):
        super(ExampleNet3, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5)
        self.batchnorm_1 = nn.BatchNorm2d(4)
        self.conv2d_2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5)
        self.batchnorm_2 = nn.BatchNorm2d(4)
        self.dropout_1 = nn.Dropout2d(p=.4)

        self.conv2d_3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3)
        self.batchnorm_3 = nn.BatchNorm2d(8)
        self.conv2d_4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3)
        self.batchnorm_4 = nn.BatchNorm2d(8)
        self.dropout_2 = nn.Dropout2d(p=.4)

        self.conv2d_5 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.batchnorm_5 = nn.BatchNorm2d(16)
        self.conv2d_6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        self.batchnorm_6 = nn.BatchNorm2d(16)
        self.dropout_3 = nn.Dropout2d()

        self.linear1 = nn.Linear(in_features=16*2*2,out_features=32)
        self.batchnorm_7 = nn.BatchNorm1d(32)
        self.dropout_4 = nn.Dropout2d(p=.6)

        self.linear2 = nn.Linear(in_features=32, out_features=24)

        self.linear3 = nn.Linear(in_features=24, out_features=7)

    def forward(self, x):
        # (1) input layer
        x = x

        # (2) hidden conv layer (block 1)
        x = self.conv2d_1(x)
        x = F.elu(x)
        x = self.batchnorm_1(x)

        # (3) hidden conv layer (block 1)
        x = self.conv2d_2(x)
        x = F.elu(x)
        x = self.batchnorm_2(x)

        # (4) end block 1
        x = F.max_pool2d(x, kernel_size=2)
        x = self.dropout_1(x)

        # (5) hidden conv layer (block 2)
        x = self.conv2d_3(x)
        x = F.elu(x)
        x = self.batchnorm_3(x)

        # (6) hidden conv layer (block 2)
        x = self.conv2d_4(x)
        x = F.elu(x)
        x = self.batchnorm_4(x)

        # (7) end block 2
        x = F.max_pool2d(x, kernel_size=2)
        x = self.dropout_2(x)

        # (8) hidden conv layer (block 3)
        x = self.conv2d_5(x)
        x = F.elu(x)
        x = self.batchnorm_5(x)

        # (9) hidden conv layer (block 3)
        x = self.conv2d_6(x)
        x = F.elu(x)
        x = self.batchnorm_6(x)

        # (10) end block 3
        x = F.max_pool2d(x, kernel_size=2)
        x = self.dropout_3(x)

        # (11) Linear layer
        x = x.reshape(-1, 16*2*2)
        x = self.linear1(x)
        x = self.batchnorm_7(x)
        x = self.dropout_4(x)

        # (12) Linear layer
        x = self.linear2(x)
        x = F.relu(x)

        # (13) Linear Layer
        x = self.linear3(x)
        x = F.softmax(x, dim=1)

        return x
