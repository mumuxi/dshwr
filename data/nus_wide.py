import os

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from data.transform import train_transform, query_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(root, num_query, num_train, batch_size, num_workers):
    """
    Loading nus-wide dataset.

    Args:
        tc(int): Top class.
        root(str): Path of image files.
        num_query(int): Number of query data.
        num_train(int): Number of training data.
        batch_size(int): Batch size.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, train_dataloader, retrieval_dataloader(torch.evaluate.data.DataLoader): Data loader.
    """
    query_dataset = NusWideDatasetTC10(
        root,
        'test_img.txt',
        'test_label.txt',
        transform=query_transform(),
    )

    train_dataset = NusWideDatasetTC10(
        root,
        'database_img.txt',
        'database_label.txt',
        transform=train_transform(),
        train=True,
        num_train=num_train,
    )

    retrieval_dataset = NusWideDatasetTC10(
        root,
        'database_img.txt',
        'database_label.txt',
        transform=query_transform(),
    )

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=num_workers,
    )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers,
    )

    return query_dataloader, train_dataloader, retrieval_dataloader

class NusWideDatasetTC10(Dataset):
    """
    Nus-wide dataset, 21 classes.

    Args
        root(str): Path of image files.
        img_txt(str): Path of txt file containing image file name.
        label_txt(str): Path of txt file containing image label.
        transform(callable, optional): Transform images.
        train(bool, optional): Return training dataset.
        num_train(int, optional): Number of training data.
    """
    def __init__(self, root, img_txt, label_txt, transform=None, train=None, num_train=None):
        self.root = root
        self.transform = transform
        self.num_classes = 10

        img_txt_path = os.path.join(root, img_txt)
        label_txt_path = os.path.join(root, label_txt)

        # Read files
        with open(img_txt_path, 'r') as f:
            self.data = np.array([i.strip() for i in f])
        self.targets = np.loadtxt(label_txt_path, dtype=np.float32)

        # Sample training dataset
        if train is True:
            perm_index = np.random.permutation(len(self.data))[:num_train]
            self.data = self.data[perm_index]
            self.targets = self.targets[perm_index]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.data[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[index], index

    def __len__(self):
        return len(self.data)

    def get_targets(self):
        return torch.from_numpy(self.targets).float()

    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()

