import torch
from torch import nn
from torchvision import datasets, models
from torchvision.datasets import FashionMNIST

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10


def main():
    net = models.resnet50()


if __name__ == "__main__":
    main()
