import torch.nn as nn
import torch.nn.functional as F
import torch

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

class ExampleNet3(nn.Module):
    def __init__(self):
        super(ExampleNet3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5),
            nn.BatchNorm2d(4),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=.4)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=.4)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Dropout2d()
        )

        self.layer4 = nn.Sequential(
            nn.Linear(in_features=16*2*2,out_features=32),
            nn.BatchNorm1d(32),
            nn.Dropout2d(p=.6)
        )

        self.output = nn.Linear(in_features=32, out_features=7)

    def forward(self, x):
        # (1) Layer 1 - Conv, batchnorm, conv, batchnorm, dropout
        x = self.layer1(x)

        # (2) Layer 2 - Conv, batchnorm, conv, batchnorm, dropout
        x = self.layer2(x)

        # (3) Layer 3 - Conv, batchnorm, conv, batchnorm, dropout
        x = self.layer3(x)

        # (4) Layer 4 - Linear, batch norm, dropout
        x = x.reshape(x.size(0),-1)
        x = self.layer4(x)

        # (5) output
        x = self.output(x)
        x = F.softmax(x, dim=1)

        return x

class JNet1(nn.Module):
    def __init__(self):
        super(JNet1, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(1,10, kernel_size=5),
                                    nn.BatchNorm2d(10),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(nn.Conv2d(10,20, kernel_size=4),
                                    nn.BatchNorm2d(20),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))

        self.fc = nn.Linear(1620, 7)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0),-1)

        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return(x)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CanNet(nn.Module):
    def __init__(self):
        super(CanNet, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(1,10, kernel_size=5),
                                    nn.BatchNorm2d(10),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(nn.Conv2d(10,20, kernel_size=4),
                                    nn.BatchNorm2d(20),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))

        self.fc = nn.Linear(1620, 7)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0),-1)

        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return(x)

class JNet2(nn.Module):
    def __init__(self):
        super(JNet2, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(1,10, kernel_size=5),
                                    nn.BatchNorm2d(10),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(nn.Conv2d(10,20, kernel_size=4),
                                    nn.BatchNorm2d(20),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))

        self.layer3 = nn.Sequential(nn.Conv2d(20,25, kernel_size=4),
                                    nn.BatchNorm2d(25),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))

        self.fc = nn.Linear(225, 7)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(x.size(0),-1)

        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return(x)

class JNet3(nn.Module):
    def __init__(self):
        super(JNet3, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(1,10, kernel_size=5),
                                    nn.BatchNorm2d(10),
                                    nn.ELU(),
                                    nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(nn.Conv2d(10,20, kernel_size=4),
                                    nn.BatchNorm2d(20),
                                    nn.ELU(),
                                    nn.MaxPool2d(kernel_size=2))

        self.fc = nn.Linear(1620, 7)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0),-1)

        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return(x)

class JNet4(nn.Module):
    def __init__(self):
        super(JNet4, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(1,10, kernel_size=11),
                                    nn.BatchNorm2d(10),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(nn.Conv2d(10,20, kernel_size=7),
                                    nn.BatchNorm2d(20),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))

        self.fc = nn.Linear(720, 7)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0),-1)

        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return(x)

class JNet5(nn.Module):
    def __init__(self):
        super(JNet5, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(1,10, kernel_size=5),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(nn.Conv2d(10,20, kernel_size=4),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))

        self.fc = nn.Linear(1620, 7)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0),-1)

        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return(x)

class JNet6(nn.Module):
    def __init__(self):
        super(JNet6, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(1,3, kernel_size=5),
                                    nn.BatchNorm2d(3),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(nn.Conv2d(3,5, kernel_size=4),
                                    nn.BatchNorm2d(5),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))

        self.fc = nn.Linear(405, 7)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0),-1)

        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return(x)

class JNet7(nn.Module):
    def __init__(self):
        super(JNet7, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(1,25, kernel_size=5),
                                    nn.BatchNorm2d(25),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(nn.Conv2d(25,64, kernel_size=4),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))

        self.fc = nn.Linear(5184, 7)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0),-1)

        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return(x)

class JNet8(nn.Module):
    def __init__(self):
        super(JNet8, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(1,10, kernel_size=5),
                                    nn.BatchNorm2d(10),
                                    nn.ReLU())

        self.layer2 = nn.Sequential(nn.Conv2d(10,20, kernel_size=4),
                                    nn.BatchNorm2d(20),
                                    nn.ReLU())

        self.fc = nn.Linear(33620, 7)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0),-1)

        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return(x)

class JNet9(nn.Module):
    def __init__(self):
        super(JNet9, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(1,10, kernel_size=5),
                                    nn.BatchNorm2d(10),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(nn.Conv2d(10,20, kernel_size=4),
                                    nn.BatchNorm2d(20),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))

        self.fc = nn.Sequential(nn.Linear(1620, 512),
                                nn.ReLU(),
                                nn.Linear(512, 7))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0),-1)

        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return(x)

class JNet10(nn.Module):
    def __init__(self):
        super(JNet10, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(1,10, kernel_size=5),
                                    nn.BatchNorm2d(10),
                                    nn.Sigmoid(),
                                    nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(nn.Conv2d(10,20, kernel_size=4),
                                    nn.BatchNorm2d(20),
                                    nn.Sigmoid(),
                                    nn.MaxPool2d(kernel_size=2))

        self.fc = nn.Linear(1620, 7)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0),-1)

        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return(x)

class JNet11(nn.Module):
    def __init__(self):
        super(JNet11, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(1,25, kernel_size=5),
                                    nn.BatchNorm2d(25),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(nn.Conv2d(25,64, kernel_size=4),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))

        self.fc = nn.Sequential(nn.Linear(5184, 2048),
                                nn.ReLU(),
                                nn.Linear(2048,7))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0),-1)

        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return(x)

