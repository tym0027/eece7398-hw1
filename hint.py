
####################################################
# First of the first, please start writing it early!
####################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision


# ----------------- prepare training data -----------------------
train_data = torchvision.datasets.CIFAR10(
    root='./data.cifar10',                          # location of the dataset
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to torch.FloatTensor of shape (C x H x W)
    download=True                                   # if you haven't had the dataset, this will automatically download it for you
)

train_loader = Data.DataLoader(dataset=train_data, ...... )

# ----------------- prepare testing data -----------------------
test_data = torchvision.datasets.CIFAR10(root='./data.cifar10/', train=False, transform=torchvision.transforms.ToTensor())

test_loader = Data.DataLoader(dataset=test_data,  ...... )

# ----------------- build the model ------------------------
class My_XXXnet(nn.Module):
    def __init__(self):
        super(My_XXXnet, self).__init__()
        ...
        ...

    def forward(self, ...):
        ...
        ...
        return ...

model = My_XXXnet()
loss_func = ...
optimizer = ...

for epoch in range(...):
    for step, (input, target) in enumerate(train_loader):
        model.train()   # set the model in training mode
            ...
            ...

        if step % 50 == 0:
            test()
            ...
            ...
            save_model()
            ...
            ...


# ------ maybe some helper functions -----------
def test(......):
    model.eval()  # switch the model to evaluation mode
    ...
    ...
    print('Test set: Epoch[{}]:Step[{}] Accuracy: {}% ......'.format(...))


def save_model(......):
    ...
    torch.save(model.state_dict(), "./model/xxxx.pt".format(...))
    ...
    # when get a better model, you can delete the previous one
    os.remove(......)   # you need to 'import os' first
    ...


def load_model(model, ......):
    ...
    model.load_state_dict(torch.load("./model/xxxx.pt"))
    ...


## parse the argument e.x. >>> python3 classify.py train
import argparse
parser = argparse.ArgumentParser(description='xxx')
parser.add_argument('mode', type=str, default="test", help="working in training/testing mode")
args = parser.parse_args()

if  args.mode == "train":
    print("train")
else:
    print("test")


def main():
    ...

if __name__ == "__main__":
    main()
