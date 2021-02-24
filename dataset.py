import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def imShow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor()]
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
    
    # for data in iter(trainloader):
    #     images, labels = data
    #     imShow(torchvision.utils.make_grid(images))

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')