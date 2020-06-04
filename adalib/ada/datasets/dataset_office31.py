"""
Dataset setting and data loader for Office31.
See `Domain Adaptation Project at Berkeley <https://people.eecs.berkeley.edu/~jhoffman/domainadapt/#datasets_code>`_. 
"""
import tarfile
import os
import pickle
import urllib
import glob
from PIL import Image

import numpy as np
import torch
import torch.utils.data as data

from sklearn import preprocessing


class Office31(data.Dataset):
    """Office31 Domain Adaptation Dataset from the 
    `Domain Adaptation Project at Berkeley <https://people.eecs.berkeley.edu/~jhoffman/domainadapt/#datasets_code>`_. 

    Args:
        root (string): 
            Root directory of dataset where dataset file exist.
        train (bool, optional): 
            If True, resample from dataset randomly.
        download (bool, optional):
            If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    # TODO: find a downloadable url
    url = "https://drive.google.com/open?id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE"

    def __init__(self, root, domain=None, train=True, transform=None, download=False):
        """Init Office31 dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.filename = "domain_adaptation_images.tar.gz"
        self.dirname = "Office31"
        self.train = train

        self.transform = transform
        self.dataset_size = None
        self.domain = domain

        # download dataset.
        # if download:
        #     self.download()
        # if not self._check_exists():
        #     raise RuntimeError(
        #         "Dataset not found." + " You can use download=True to download it"
        #     )

        self.labeler = preprocessing.LabelEncoder()
        self.data, self.targets = self.load_samples()
        self.targets = torch.LongTensor(self.targets)

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        path = os.path.join(self.root, self.dirname, self.domain, self.data[index])
        img = None
        with open(path, "rb") as f:
            with Image.open(f) as imgf:
                img = imgf.convert("RGB")

        label = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([np.int64(label).item()])
        # label = torch.FloatTensor([label.item()])
        return img, label

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.filename))

    def download(self):
        """Download dataset."""
        raise NotImplementedError(f"Please download manually from {self.url}")
        logging.info("[DONE]")
        return

    def extract_gzip(self):
        raise NotImplementedError(f"Please detar manually from {self.filename}")
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.join(self.root, self.dirname)
        with tarfile.open(filename, "r:gz") as tar:
            for member in tar.getmembers():
                f = tar.extractfile(member)
                if f is not None:
                    content = f.read()

    def load_samples(self):
        """Load sample images from dataset."""
        imgdir = os.path.join(self.root, self.dirname, self.domain, "images")
        image_list = glob.glob(f"{imgdir}/*/*.jpg")
        labels = [os.path.split(os.path.split(p)[0])[-1] for p in image_list]
        labels = self.labeler.fit_transform(labels)
        n_total = len(image_list)
        n_test = int(0.1 * n_total)
        indices = np.arange(n_total)
        rg = np.random.RandomState(seed=128753)
        rg.shuffle(indices)
        train_indices = indices[:-n_test]
        test_indices = indices[-n_test:]
        if self.train:
            images = np.array(image_list)[train_indices].tolist()
            labels = np.array(labels)[train_indices].tolist()
            self.dataset_size = len(images)
            self.labeler
        else:
            images = np.array(image_list)[test_indices].tolist()
            labels = np.array(labels)[test_indices].tolist()
            self.dataset_size = len(images)
        return images, labels
