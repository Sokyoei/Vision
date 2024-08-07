import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from Vision.utils import DEVICE

EPOCHS = 10
BATCH_SIZE = 64
NUM_WORKERS = 1


class LeNet5(nn.Module):

    def __init__(self, n_classes) -> None:
        super().__init__()
        # Nx1x32x32
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 5, 1, 2), nn.BatchNorm2d(6), nn.ReLU())
        # Nx6x28x28
        self.pool1 = nn.MaxPool2d(2, 2, 0)
        # Nx6x14x14
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5, 1, 0), nn.BatchNorm2d(16), nn.ReLU())
        # Nx16x10x10
        self.pool2 = nn.MaxPool2d(2, 2, 0)
        # Nx16x5x5
        self.fc1 = nn.Sequential(nn.Linear(16 * 5 * 5, 120), nn.ReLU())
        # 120
        self.fc2 = nn.Sequential(nn.Linear(120, 84), nn.ReLU())
        # 84
        self.fc3 = nn.Linear(84, n_classes)
        # n_classes

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        x4 = self.pool2(x3)
        x4: Tensor = x4.reshape(x4.size(0), -1)
        x5 = self.fc1(x4)
        x6 = self.fc2(x5)
        x7 = self.fc3(x6)
        return x7


def mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    train_images = datasets.MNIST("./", True, transform, download=True)
    test_images = datasets.MNIST("./", False, transform, download=True)
    train_data = DataLoader(train_images, BATCH_SIZE, True, num_workers=NUM_WORKERS)
    test_data = DataLoader(test_images, BATCH_SIZE, num_workers=NUM_WORKERS)
    return train_data, test_data


train_data, test_data = mnist()


def train():
    net = LeNet5(10).to(DEVICE)
    losser = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.005)

    for i in range(EPOCHS):
        sum_loss = 0
        for X_train, y_train in train_data:
            X_train: Tensor
            y_train: Tensor
            X_train = X_train.to(DEVICE)
            y_train = y_train.to(DEVICE)
            optimizer.zero_grad()
            y_pred = net(X_train)
            loss: Tensor = losser(y_pred, y_train)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()

        # test
        total = 0
        correct = 0
        for X_test, y_test in test_data:
            X_test: Tensor
            y_test: Tensor
            X_test = X_test.to(DEVICE)
            y_test = y_test.to(DEVICE)
            y_pred: Tensor = net(X_test)
            _, predict = torch.max(y_pred.data, 1)
            total += y_test.size(0)
            correct += (predict == y_test).sum()

        print(f"epoch: {i + 1}, train_loss: {sum_loss / len(train_data)}, test_acc: {correct / total}")

    torch.save(net.state_dict(), "LeNet5.pt")


def main():
    train()


if __name__ == "__main__":
    main()
