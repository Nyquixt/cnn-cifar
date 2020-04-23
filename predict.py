import torch
import torchvision
import torchvision.transforms as transforms
from model import VGG
import matplotlib.pyplot as plt
import numpy as np

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

model = VGG('VGG11')
model = torch.load('vgg11-cifar10.chkpt')

# functions to show an image


def imshow(img):
    # unnormalize
    img = img / 2 + 0.5  
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

model.to('cpu')

for i, batch in enumerate(testloader):
    imgs, labels = batch
    imshow(torchvision.utils.make_grid(imgs))
    preds = model(imgs)
    preds = preds.detach().numpy()
    print([classes[np.argmax(x)] for x in preds])
    if (i == 2):
      break