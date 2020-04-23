import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import argparse
import time

from model import VGG

parser = argparse.ArgumentParser(description='Training VGG16 on CIFAR10')

parser.add_argument('--epoch', '-e', type=int, default=30, help='Number of epochs')
parser.add_argument('--batch', '-b', type=int, default=4, help='The batch size')
parser.add_argument('--lr', '-l', type=float, default=0.001, help='Learning rate')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum for SGD')
parser.add_argument('--update', '-u', type=int, default=2000, help='Print out stats after x batches')

args = parser.parse_args()
print(args)

# define some hyper-params
n_epoch = args.epoch
batch_size = args.batch
lr = args.lr
momentum = args.momentum
every_batch = args.update

# define transform for images
# data augmentation
transform = transforms.Compose(
    [transforms.RandomCrop(size=32, padding=4),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomRotation(degrees=45),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # this is the std and mean from ImageNet
    ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# define losses lists to plot
train_losses = []
val_losses = []

# define model
vgg = VGG('VGG16')
# vgg = torch.load('vgg-cifar10.chkpt')
vgg.cuda()
vgg.train()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(vgg.parameters(), lr=lr, momentum=momentum)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.95)

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
        outputs = vgg(inputs)
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

                    outputs = vgg(inputs)
                    loss = criterion(outputs, labels)

                    validation_loss += loss.item()
            
            train_losses.append(training_loss / every_batch)
            val_losses.append(validation_loss / (10000/batch_size))

            print('[Epoch: %d, Batch: %5d] Train Loss: %.3f    Val Loss: %.3f' %
                  ( epoch + 1, i + 1, training_loss / every_batch, validation_loss / (10000/batch_size) ))
            
            training_loss = 0.0

    # step the scheduler after every epoch
    scheduler.step()

# Test the model
vgg.eval()
vgg.to('cpu')

with torch.no_grad():
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = vgg(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the vgg on the 10000 test images: {} %'.format((correct / total) * 100))

# Save the model
torch.save(vgg, 'vgg-cifar10-b{}-e{}-{}.chkpt'.format(batch_size, n_epoch, int(round(time.time() * 1000))))

# Save plot
x = np.array([x for x in range(len(train_losses))]) * every_batch
y1 = np.array(train_losses)

y2 = np.array(val_losses)

plt.plot(x, y1, label='train loss')
plt.plot(x, y2, label='val loss')
plt.legend()
plt.xlabel('batches')
plt.ylabel('losses')
plt.savefig('losses-b{}-e{}-{}.png'.format(batch_size, n_epoch, int(round(time.time() * 1000))))