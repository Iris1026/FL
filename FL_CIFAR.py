import torch
from torchvision import datasets, transforms
import torch.utils.data.dataloader as dataloader
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 10
num_epochs = 5
batch_size = 100

train_dataset = datasets.CIFAR10(root='./dataset',train=True,transform=transforms.ToTensor(),download=True)
train_loader = dataloader.DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)


# test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)
test_dataset = Subset(train_dataset,range(0,2000))
test_loader = dataloader.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def print_model_parameters(model):
    for name, param in model.named_parameters():
        print(name, param.size())

def dirichlet_split_noniid(train_labels, alpha, n_clients):
    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(k_idcs, (np.cumsum(fracs)[:-1]*len(k_idcs)).astype(int))):
            client_idcs[i] += [idcs]
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    # Calculate weights based on the number of data points for each client
    client_weights = [len(idcs) for idcs in client_idcs]

    return client_idcs,client_weights


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)  # Reshape the input shape to (batch_size, 1, 28, 28).
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the feature map
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Client update function
def client_update(client_model, optimizer, train_loader, epoch):

    for e in range(epoch):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 32 * 32).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = client_model(images)

            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss.item()

# Combine client models
def combine_params(client_models,client_weights):

    # Normalize weights
    total_weight = sum(client_weights)
    normalized_weights = [w / total_weight for w in client_weights]
    combined_params = []
    for param_i in zip(*[model.parameters() for model in client_models]):
        weighted_avg_param = sum(w * param for w, param in zip(normalized_weights, param_i))
        combined_params.append(weighted_avg_param)

    return combined_params

def evaluate(globel_model, test_loader):

    total_loss = 0
    correct = 0
    total = len(test_loader.dataset)
    for i, (images, labels) in enumerate(test_loader):
        images = images.reshape(-1,32*32).to(device)
        labels = labels.to(device)

        outputs = globel_model(images)
        total_loss += criterion(outputs, labels)

        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()

    print("correct={}".format(correct))
    print("total = {}".format(total))
    average_loss = total_loss/total
    accuracy = (correct/total)*100

    print("The average loss is {}. The accuracy of the model is {}%".format(average_loss, accuracy))


if __name__ == "__main__":


    global_model = NeuralNet().to(device) # initialize the global model

    optimizer = optim.SGD(global_model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    num_clients = 3

    # Create client models
    client_models = [NeuralNet().to(device) for _ in range(num_clients)]

    optimizers = [optim.Adam(model.parameters(), lr=0.01) for model in client_models]

    # Split dataset among clients using non-IID method
    alpha = 0.5  # this parameter can be adjusted
    train_labels = np.array([item[1] for item in train_dataset])
    client_idcs, client_weights = dirichlet_split_noniid(train_labels, alpha, num_clients)
    client_datasets = [Subset(train_dataset, idcs) for idcs in client_idcs]
    client_loaders = [dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in
                      client_datasets]

    # Federated learning loop
    for round in range(num_epochs):
        # Client update
        for model, optimizer, loader in zip(client_models, optimizers, client_loaders):
            client_update(model,optimizer, loader, epoch=5)

        # Aggregate the parameters of client models
        global_params = combine_params(client_models,client_weights)

        # Update global model
        global_model.load_state_dict({name: param for name, param in zip(global_model.state_dict().keys(), global_params)})


        # Distribute global model to clients
        for model in client_models:
            model.load_state_dict(global_model.state_dict())

    evaluate(global_model, test_loader)
