import unittest

import ada.datasets.digits_dataset_access as dda
import ada.datasets.preprocessing as pp


class TestDigitDataset(unittest.TestCase):
    def check_data_access(self, data_access, expected_class, expected_transform_kind):
        self.assertEqual(expected_class, data_access.__class__)
        self.assertEqual("/one/path/", data_access._data_path)
        self.assertEqual(
            repr(pp.get_transform(expected_transform_kind)),
            repr(data_access._transform),
        )

    def test_mnist_to_usps(self):
        source, target, n_channels = dda.DigitDataset.get_accesses(
            dda.DigitDataset.MNIST, dda.DigitDataset.USPS, "/one/path/"
        )
        self.assertEqual(1, n_channels)
        self.check_data_access(source, dda.MNISTDatasetAccess, "mnist32")
        self.check_data_access(target, dda.USPSDatasetAccess, "usps32")

    def test_mnist_to_svhn(self):
        source, target, n_channels = dda.DigitDataset.get_accesses(
            dda.DigitDataset.MNIST, dda.DigitDataset.SVHN, "/one/path/"
        )
        self.assertEqual(3, n_channels)
        self.check_data_access(source, dda.MNISTDatasetAccess, "mnist32rgb")
        self.check_data_access(target, dda.SVHNDatasetAccess, "svhn")

    def test_mnist_to_mnistm(self):
        source, target, n_channels = dda.DigitDataset.get_accesses(
            dda.DigitDataset.MNIST, dda.DigitDataset.MNISTM, "/one/path/"
        )
        self.assertEqual(3, n_channels)
        self.check_data_access(source, dda.MNISTDatasetAccess, "mnist32rgb")
        self.check_data_access(target, dda.MNISTMDatasetAccess, "mnistm")

    def test_svhn_to_mnist(self):
        source, target, n_channels = dda.DigitDataset.get_accesses(
            dda.DigitDataset.SVHN, dda.DigitDataset.MNIST, "/one/path/"
        )
        self.assertEqual(3, n_channels)
        self.check_data_access(source, dda.SVHNDatasetAccess, "svhn")
        self.check_data_access(target, dda.MNISTDatasetAccess, "mnist32rgb")

    def test_mnistm_to_usps(self):
        source, target, n_channels = dda.DigitDataset.get_accesses(
            dda.DigitDataset.MNISTM, dda.DigitDataset.USPS, "/one/path/"
        )
        self.assertEqual(3, n_channels)
        self.check_data_access(source, dda.MNISTMDatasetAccess, "mnistm")
        self.check_data_access(target, dda.USPSDatasetAccess, "usps32rgb")
