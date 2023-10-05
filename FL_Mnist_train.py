import torch
from torchvision import datasets, transforms
import torch.utils.data.dataloader as dataloader
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import numpy as np

train_set = datasets.MNIST(root="./", train=True, transform=transforms.ToTensor(), download=True)
test_set = datasets.MNIST(root="./", train=False, transform=transforms.ToTensor(), download=True)

test_set = Subset(test_set, range(0,2000))

train_set_A = Subset(train_set, range(0,20000))
train_set_B = Subset(train_set, range(20000,40000))
train_set_C = Subset(train_set, range(40000,60000))

# train_loader = dataloader.DataLoader(dataset=train_set, batch_size=1, shuffle=True)
test_loader = dataloader.DataLoader(dataset=test_set, batch_size=1, shuffle=False)
train_loader_A = dataloader.DataLoader(dataset=train_set_A, batch_size = 1000, shuffle=True)
train_loader_B = dataloader.DataLoader(dataset=train_set_B, batch_size=1000, shuffle=True)
train_loader_C = dataloader.DataLoader(dataset=train_set_C, batch_size=1000, shuffle=True)


def print_model_parameters(model):
    for name, param in model.named_parameters():
        print(name, param.size())



# 普通的训练测试过程
def train_and_test_1(train_loader, test_loader):
    class NeuralNet(nn.Module):
        def __init__(self):
            super(NeuralNet, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*7*7, 128)
            self.fc2 = nn.Linear(128, 10)
            nn.init.normal_(self.fc1.weight)     # 使用正态分布初始化参数
            nn.init.normal_(self.fc2.weight)
            nn.init.normal_(self.conv1.weight)     # 使用正态分布初始化参数
            nn.init.normal_(self.conv2.weight)
            nn.init.constant_(self.fc1.bias, val=0)
            nn.init.constant_(self.fc2.bias, val=0)

        def forward(self, x):
            x = x.view(-1, 1, 28, 28)  # 调整输入形状为(batch_size, 1, 28, 28)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)  # 展平特征图
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    epoches = 5
    lr = 0.01

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeuralNet().to(device)

    # print_model_parameters(model)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

# 训练
    for epoch in range(epoches):
        i=0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)

            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i += 1

    params = list(model.named_parameters()) # 获取模型参数
    # print(params)

# 测试

    correct = 0
    total = len(test_loader.dataset)
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()


    print("The accuracy is {}%".format(correct/total*100))

    return params


# 联邦后的训练测试过程
def train_and_test_2(train_loader, test_loader, com_conv1_w, com_conv2_w, com_fc1_w, com_fc2_w):
    class NeuralNet(nn.Module):
        def __init__(self, com_conv1_w, com_conv2_w, com_fc1_w, com_fc2_w):
            super(NeuralNet,self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*7*7, 128)
            self.fc2 = nn.Linear(128, 10)
            self.fc1.weight = Parameter(com_fc1_w)
            self.fc2.weight = Parameter(com_fc2_w)
            self.conv1.weight = Parameter(com_conv1_w)
            self.conv2.weight = Parameter(com_conv2_w)
            nn.init.constant_(self.fc1.bias, val=0)
            nn.init.constant_(self.fc2.bias, val=0)

        def forward(self, x):
            x = x.view(-1, 1, 28, 28)  # 调整输入形状为(batch_size, 1, 28, 28)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)  # 展平特征图
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    epoches = 5
    lr = 0.01

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeuralNet(com_conv1_w, com_conv2_w, com_fc1_w, com_fc2_w).to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

# 训练
    for epoch in range(epoches):
        i=0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)

            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i += 1

    params = list(model.named_parameters()) # 获取模型参数

# 测试

    correct = 0
    total = len(test_loader.dataset)
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)

        pred = output.argmax(dim=1, keepdim=True)
        print(labels.size())
        correct += pred.eq(labels.view_as(pred)).sum().item()


    print("The accuracy is {}%".format(correct/total*100))

    return params


def combine_params(para_A, para_B, para_C):
    # print(para_A)

    conv1_wA = para_A[0][1].data
    conv1_wB = para_B[0][1].data
    conv1_wC = para_C[0][1].data

    conv2_wA = para_A[2][1].data
    conv2_wB = para_B[2][1].data
    conv2_wC = para_C[2][1].data


    fc1_wA = para_A[4][1].data
    fc1_wB = para_B[4][1].data
    fc1_wC = para_C[4][1].data

    fc2_wA = para_A[6][1].data
    fc2_wB = para_B[6][1].data
    fc2_wC = para_C[6][1].data


    com_conv1_w = (conv1_wA + conv1_wB + conv1_wC) / 3
    com_conv2_w = (conv2_wA + conv2_wB + conv2_wC) / 3
    com_fc1_w = (fc1_wA + fc1_wB + fc1_wC) / 3
    com_fc2_w = (fc2_wA + fc2_wB + fc2_wC) / 3

    return com_conv1_w, com_conv2_w, com_fc1_w, com_fc2_w


if __name__ == "__main__":


    para_A = train_and_test_1(train_loader_A, test_loader)
    para_B = train_and_test_1(train_loader_B, test_loader)
    para_C = train_and_test_1(train_loader_C, test_loader)

    for i in range(5):
        print("The {} round to be federated".format(i+1))
        com_conv1_w, com_conv2_w, com_fc1_w, com_fc2_w = combine_params(para_A,para_B,para_C)
        para_A = train_and_test_2(train_loader_A, test_loader, com_conv1_w, com_conv2_w, com_fc1_w, com_fc2_w)
        para_B = train_and_test_2(train_loader_B, test_loader, com_conv1_w, com_conv2_w, com_fc1_w, com_fc2_w)
        para_C = train_and_test_2(train_loader_C, test_loader, com_conv1_w, com_conv2_w, com_fc1_w, com_fc2_w)

