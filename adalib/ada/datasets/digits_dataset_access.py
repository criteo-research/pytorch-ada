from enum import Enum
from torchvision.datasets import MNIST, SVHN
import ada.datasets.preprocessing as proc
from ada.datasets.dataset_usps import USPS
from ada.datasets.dataset_mnistm import MNISTM
from ada.datasets.dataset_access import DatasetAccess


class DigitDataset(Enum):
    MNIST = "MNIST"
    MNISTM = "MNISTM"
    USPS = "USPS"
    SVHN = "SVHN"

    @staticmethod
    def get_accesses(source: "DigitDataset", target: "DigitDataset", data_path):
        transforms_default_names = {
            DigitDataset.MNIST: "mnist32",
            DigitDataset.MNISTM: "mnistm",
            DigitDataset.USPS: "usps32",
            DigitDataset.SVHN: "svhn",
        }

        factories = {
            DigitDataset.MNIST: MNISTDatasetAccess,
            DigitDataset.MNISTM: MNISTMDatasetAccess,
            DigitDataset.USPS: USPSDatasetAccess,
            DigitDataset.SVHN: SVHNDatasetAccess,
        }

        source_tf = transforms_default_names[source]
        target_tf = transforms_default_names[target]
        num_channels = 1

        # handle color/nb channels
        if source is DigitDataset.MNIST and target in [
            DigitDataset.SVHN,
            DigitDataset.MNISTM,
        ]:
            source_tf = "mnist32rgb"
            num_channels = 3
        if target is DigitDataset.MNIST and source in [
            DigitDataset.SVHN,
            DigitDataset.MNISTM,
        ]:
            target_tf = "mnist32rgb"
            num_channels = 3

        # TODO: what about USPS?

        return (
            factories[source](data_path, source_tf),
            factories[target](data_path, target_tf),
            num_channels,
        )


class DigitDatasetAccess(DatasetAccess):
    def __init__(self, data_path, transform_kind):
        super().__init__(n_classes=10)
        self._data_path = data_path
        self._transform = proc.get_transform(transform_kind)


class MNISTDatasetAccess(DigitDatasetAccess):
    def get_train(self):
        return MNIST(
            self._data_path, train=True, transform=self._transform, download=True
        )

    def get_test(self):
        return MNIST(
            self._data_path, train=False, transform=self._transform, download=True
        )


class MNISTMDatasetAccess(DigitDatasetAccess):
    def get_train(self):
        return MNISTM(
            self._data_path, train=True, transform=self._transform, download=True
        )

    def get_test(self):
        return MNISTM(
            self._data_path, train=False, transform=self._transform, download=True
        )


class USPSDatasetAccess(DigitDatasetAccess):
    def get_train(self):
        return USPS(
            self._data_path, train=True, transform=self._transform, download=True
        )

    def get_test(self):
        return USPS(
            self._data_path, train=False, transform=self._transform, download=True
        )


class SVHNDatasetAccess(DigitDatasetAccess):
    def get_train(self):
        return SVHN(
            self._data_path, split="train", transform=self._transform, download=True
        )

    def get_test(self):
        return SVHN(
            self._data_path, split="test", transform=self._transform, download=True
        )
