####################################################
# First of the first, please start writing it early!
####################################################
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import torch.optim

from datetime import datetime

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

'''
import argparse
parser = argparse.ArgumentParser(description='xxx')
parser.add_argument('mode', type=str, default="test", help="working in training/testing mode")
args = parser.parse_args()
'''

class timnet(nn.Module):
    def __init__(self):
        super(timnet, self).__init__()
        # My code
        self.linear1 = nn.Linear(3072, 3072)
        self.linear2 = nn.Linear(3072, 2048)
        self.linear3 = nn.Linear(2048, 1024)
        self.linear4 = nn.Linear(1024, 512)
        self.linear5 = nn.Linear(512, 10) 
        self.activation = nn.ReLU()
        
        '''
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        '''
    def forward(self, x):
        # My code
        # print("size: ", x.shape)
        x = self.activation(self.linear1(x))
        # print("size: ", x.shape)
        x = self.activation(self.linear2(x))
        # print("size: ", x.shape)
        x = self.activation(self.linear3(x))
        # print("size: ", x.shape)
        x = self.activation(self.linear4(x))
        # print("size: ", x.shape)
        x = self.linear5(x)
        # print("size: ", x.shape)
        return x # torch.sum(x,dim=0)
        
        '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        '''
# ------ maybe some helper functions -----------
def test(model, epoch, step, acc):
    model.eval()  # switch the model to evaluation mode
    print('Test set: Epoch[{}]:Step[{}] Accuracy: {}% ......'.format(epoch, step, acc))


def saveModel(model, path):
    torch.save(model.state_dict(), path)

def loadModel(model, path):
    model.load_state_dict(torch.load(path))
    return model

try:
    if  sys.argv[1] == "train":
        print("train")
    elif sys.argv[1] == "test":
        print("test")
except IndexError:
    print("train (default)")

def main():
    train_data = torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=16, shuffle=True, num_workers=8)

    test_data = torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=False, transform=torchvision.transforms.ToTensor())
    test_loader = Data.DataLoader(dataset=test_data)

    model = timnet().cuda(0)
    criterion = nn.CrossEntropyLoss()
    # loss_func = torch.nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    start_epoch = 0
    total_epochs = 10
    for epoch in range(start_epoch, total_epochs):
        # Train
        model.train(True)
        optimizer.zero_grad()
        total_loss = 0
        total_acc = 0
        data_length = 0 # len(enumerate(train_loader))
        for step, (input, target) in enumerate(train_loader):
            # print(input.shape, target.shape)
            predictions = model.forward(input.view(16, -1).cuda(0))
            loss = criterion(predictions, target.cuda(0))
            total_loss += loss.item()
            
            values, indices = torch.max(predictions, 1)
            
            # print(indices,":",values, " : ", target)

            for i in range(0,16):
                # print(indices[i],target[i])
                if indices[i] == target[i]:
                    # print("Success!!!")
                    total_acc += 1;

            loss.backward()
            optimizer.step()
            data_length += 16
        
        dateTimeObj = datetime.now()
        timestamp = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")

        print(str(timestamp) + ": (" + str(epoch) + ") Trainig Loss: " + str(total_loss/data_length) + " | Training Accuracy: " + str(100*(total_acc/data_length)))

        # Validate
        model.train(False)
        model.eval()
        optimizer.zero_grad()
        total_loss = 0
        total_acc = 0
        data_length = 0 # len(enumerate(train_loader))
        for step, (input, target) in enumerate(test_loader):
            predictions = model(input.view(-1).cuda(0))
            loss = criterion(predictions.unsqueeze(0), target.cuda(0))
            total_loss += loss.item()

            values, indices = torch.max(predictions, 0)

            # print(indices.item(),":",values.item(), " : ", target.item())

            if indices.item() == target.item():
                # print("Success!!!")
                total_acc += 1;

            # loss.backward()
            # optimizer.step()
            data_length += 1

        dateTimeObj = datetime.now()
        timestamp = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")

        print(str(timestamp) + ": (" + str(epoch) + ") Validation Loss: " + str(total_loss/data_length) + " | Validation Accuracy: " + str(100*(total_acc/data_length)))

        saveModel(model, "./classify_cifar10_" + str(epoch) + ".pth")

if __name__ == "__main__":
    main()
