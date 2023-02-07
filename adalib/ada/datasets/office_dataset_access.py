from enum import Enum
import ada.datasets.preprocessing as proc
from ada.datasets.dataset_access import DatasetAccess
from ada.datasets.dataset_office31 import Office31


class Office31Dataset(Enum):
    Amazon = "amazon"
    DSLR = "dslr"
    Webcam = "webcam"

    @staticmethod
    def get_accesses(source: "Office31Dataset", target: "Office31Dataset", data_path):
        return (
            Office31DatasetAccess(source, data_path),
            Office31DatasetAccess(target, data_path),
        )


class Office31DatasetAccess(DatasetAccess):
    def __init__(self, domain, data_path):
        super().__init__(n_classes=31)
        self._data_path = data_path
        self._transform = proc.get_transform("office")
        self._domain = domain.value

    def get_train(self):
        return Office31(
            self._data_path,
            domain=self._domain,
            train=True,
            transform=self._transform,
            download=True,
        )

    def get_test(self):
        return Office31(
            self._data_path,
            domain=self._domain,
            train=False,
            transform=self._transform,
            download=True,
        )
