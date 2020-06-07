import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import argparse
import time

from nets.vgg import VGG
from nets.resnet import ResNet18, ResNet50
from nets.googlenet import GoogLeNet
from nets.alexnet import AlexNet
from nets.lenet import LeNet
from utils import calculate_acc

parser = argparse.ArgumentParser(description='Training VGG16 on CIFAR10')

parser.add_argument('--network', '-n', choices=['vgg16', 'vgg19', 'resnet18', 'resnet50', 'googlenet', 'lenet', 'alexnet'], required=True)
parser.add_argument('--epoch', '-e', type=int, default=30, help='Number of epochs')
parser.add_argument('--batch', '-b', type=int, default=4, help='The batch size')
parser.add_argument('--lr', '-l', type=float, default=0.001, help='Learning rate')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum for SGD')
parser.add_argument('--update', '-u', type=int, default=2000, help='Print out stats after x batches')
parser.add_argument('--weight-decay', '-d', type=float, default=0.0, help='Weight decay for SGD optimizer')
parser.add_argument('--step-size', '-s', type=int, default=1, help='Step in learning rate scheduler')
parser.add_argument('--gamma', '-g', type=float, default=0.95, help='Gamma in learning rate scheduler')

args = parser.parse_args()
print(args)

# define some hyper-params
n_epoch = args.epoch
batch_size = args.batch
lr = args.lr
momentum = args.momentum
every_batch = args.update
weight_decay = args.weight_decay
step_size = args.step_size
gamma = args.gamma

# define transform for images
# data augmentation for train and test set
train_transform = transforms.Compose(
    [transforms.RandomCrop(size=32, padding=4),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) # this is the std and mean from CIFAR10
    ])

test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=train_transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transform, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# define losses lists to plot
train_losses = []
val_losses = []
train_accuracy = []
val_accuracy = []

# define model
if args.network == 'vgg16':
    net = VGG('VGG16')
elif args.network == 'vgg19':
    net = VGG('VGG19')
elif args.network == 'resnet18':
    net = ResNet18()
elif args.network == 'resnet50':
    net = ResNet50()
elif args.network == 'googlenet':
    net = GoogLeNet()
elif args.network == 'lenet':
    net = LeNet()
elif args.network == 'alexnet':
    net = AlexNet()

net.cuda()
net.train()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)

# Train the model
for epoch in range(n_epoch):  # loop over the dataset multiple times

    training_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

        # print statistics
        
        if i % every_batch == (every_batch - 1):    # print every every_batch mini-batches

            with torch.no_grad():
                validation_loss = 0.0
                for j, data in enumerate(testloader): # (10,000 / batch_size) batches
                    inputs, labels = data
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    validation_loss += loss.item()
            
            train_losses.append(training_loss / every_batch)
            val_losses.append(validation_loss / (10000/batch_size))

            train_acc = calculate_acc(trainloader, net)
            net.eval()
            val_acc = calculate_acc(testloader, net)
            net.train()

            train_accuracy.append(train_acc)
            val_accuracy.append(val_acc)

            print('[Epoch: %d, Batch: %5d] Train Loss: %.3f    Train Acc: %.3f%%    Val Loss: %.3f    Val Acc: %.3f%%' %
                  ( epoch + 1, i + 1, training_loss / every_batch, train_acc, validation_loss / (10000/batch_size), val_acc ))
            
            training_loss = 0.0

    # step the scheduler after every epoch
    scheduler.step()

# Test the model
net.eval()
val_acc = calculate_acc(testloader, net)
print('Test Accuracy of the network on the 10000 test images: {} %'.format(val_acc))

# Save the model
torch.save(net.state_dict(), 'models/{}-cifar10-b{}-e{}-{}.pth'.format(args.network, batch_size, n_epoch, int(round(time.time() * 1000))))


# Save plot
x = np.array([x for x in range(len(train_losses))]) * every_batch
y1 = np.array(train_losses)
y2 = np.array(val_losses)

y3 = np.array(train_accuracy)
y4 = np.array(val_accuracy)

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

ax1.plot(x, y1, label='train loss')
ax1.plot(x, y2, label='val loss')
ax1.legend()
ax1.xaxis.set_visible(False)
ax1.set_ylabel('losses')

ax2.plot(x, y3, label='train acc')
ax2.plot(x, y4, label='val acc')
ax2.legend()
ax2.set_xlabel('batches')
ax2.set_ylabel('acc')

plt.savefig('plots/{}-losses-b{}-e{}-{}.png'.format(args.network, batch_size, n_epoch, int(round(time.time() * 1000))))
