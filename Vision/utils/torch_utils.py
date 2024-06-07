from typing import Optional

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def vision_dataset(dataset: datasets.VisionDataset, batch_size: Optional[int] = 64, num_workers: Optional[int] = 1):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    train_images = dataset("./", True, transform, download=True)
    train_images = dataset("./", False, transform, download=True)

    train_data = DataLoader(train_images, batch_size, True, num_workers=num_workers)
    test_data = DataLoader(train_images, batch_size, num_workers=num_workers)
    return train_data, test_data
