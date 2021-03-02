import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def imShow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainSet = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

testSet = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# trainSet = torchvision.datasets.MNIST(
#     root='./data',
#     train=True,
#     download=True,
#     transform=transform
# )

# testSet = torchvision.datasets.MNIST(
#     root='./data',
#     train=False,
#     download=True,
#     transform=transform
# )

trainloader = torch.utils.data.DataLoader(
    trainSet,
    batch_size=4,
    shuffle=True,
    num_workers=2)
testloader = torch.utils.data.DataLoader(
    testSet,
    batch_size=4,
    shuffle=True,
    num_workers=2)
