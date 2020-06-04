import os
from enum import Enum
from pathlib import Path
from sklearn.utils import check_random_state
import numpy as np

import ada.datasets.toys as toys
import ada.datasets.digits_dataset_access as digits
import ada.datasets.office_dataset_access as office
from ada.datasets.multisource import MultiDomainDatasets, DatasetSizeType
from ada.datasets.sampler import SamplingConfig
import ada.utils.experimentation as xp


class WeightingType(Enum):
    NATURAL = "natural"
    BALANCED = "balanced"
    PRESET0 = "preset0"


class DatasetFactory:
    """This class takes a configuration dictionary
    and generates a MultiDomainDataset class
    with the appropriate data.
    """

    def __init__(self, data_config, data_path=None, download=True, n_fewshot=0):
        """
        Args:
            data_config (dict): parameters to factor the right dataset
            data_path (str, optional): where data is stored/downloaded/created.
                if no data_path given, creates one in your home/.ada
            download (bool, optional): download (or generate for toy) the data.
            n_fewshot (int, optional): Number of target samples for which the label may be used,
                for batch sampling & train/val/test splits.
        """
        self._data_config = data_config
        if data_path is None:
            self._data_path = f"{Path.home()}/.ada"
        else:
            self._data_path = data_path
        self._download = download
        self._n_fewshot = n_fewshot
        self._long_name = self._data_config["dataset_name"]
        os.makedirs(self._data_path, exist_ok=True)

    def is_semi_supervised(self):
        return self._n_fewshot is not None and self._n_fewshot > 0

    def get_multi_domain_dataset(self, random_state):
        self._create_dataset(random_state)
        return self.domain_datasets

    def get_data_args(self):
        """Returns dataset specific arguments necessary to build the network
        first returned item is number of classes
        second is a tuple of arguments to be passed to all network_factory functions.

        Returns:
            tuple: tuple containing:
                - int: the number of classes in the dataset
                - int or None: the input dimension
                - int or None: the number of channels for images
        """
        if self._data_config["dataset_group"] == "toy":
            return (
                self._data_config["cluster"]["n_clusters"],
                self._data_config["cluster"]["dim"],
                (),
            )
        if self._data_config["dataset_group"] == "digits":
            return 10, 784, (self._num_channels,)
        if self._data_config["dataset_group"] == "office31":
            return 31, None, ()

    def get_data_short_name(self):
        return self._data_config["dataset_name"]

    def get_data_long_name(self):
        return self._long_name

    def get_data_hash(self):
        return xp.param_to_hash(self._data_config)

    def _create_dataset(self, random_state):
        random_state = check_random_state(random_state)
        if self._data_config["dataset_group"] == "toy":
            src, tgt = self._create_toy_access()
        elif self._data_config["dataset_group"] == "digits":
            src, tgt = self._create_digits_access()
        elif self._data_config["dataset_group"] == "office31":
            src, tgt = self._create_office31_access()
        else:
            raise NotImplementedError(
                f"Unknown dataset type, you can need your own dataset here: {__file__}"
            )
        self._create_domain_dataset(src, tgt, random_state)

    def _create_domain_dataset(self, source_access, target_access, random_state):
        weight_type = WeightingType(self._data_config.get("weight_type", "natural"))
        size_type = DatasetSizeType(self._data_config.get("size_type", "source"))
        self._long_name = f"{self._long_name}_{weight_type.value}_{size_type.value}"

        if weight_type is WeightingType.PRESET0:
            source_sampling_config = SamplingConfig(
                class_weights=np.arange(source_access.n_classes(), 0, -1)
            )
            target_sampling_config = SamplingConfig(
                class_weights=random_state.randint(1, 4, size=target_access.n_classes())
            )
        elif weight_type is WeightingType.BALANCED:
            source_sampling_config = SamplingConfig(balance=True)
            target_sampling_config = SamplingConfig(balance=True)
        elif weight_type not in WeightingType:
            raise ValueError(f"Unknown weighting method {weight_type}.")
        else:
            source_sampling_config = SamplingConfig()
            target_sampling_config = SamplingConfig()

        self.domain_datasets = MultiDomainDatasets(
            source_access=source_access,
            target_access=target_access,
            source_sampling_config=source_sampling_config,
            target_sampling_config=target_sampling_config,
            size_type=size_type,
            n_fewshot=self._n_fewshot,
        )

    def _create_toy_access(self):
        blob_args = self._data_config["cluster"]
        shift_params = toys.get_datashift_params(**self._data_config["shift"])
        self._long_name = (
            f"blobs_{xp.param_to_str(blob_args)}_{xp.param_to_str(shift_params)}"
        )
        n_samples = self._data_config["n_samples"]
        source_access = toys.CausalBlobsDataAccess(
            data_path=self._data_path,
            transform=toys.get_datashift_params("no_shift"),
            download=self._download,
            cluster_params=blob_args,
            n_samples=n_samples,
        )
        target_access = toys.CausalBlobsDataAccess(
            data_path=self._data_path,
            transform=shift_params,
            download=self._download,
            cluster_params=blob_args,
            n_samples=n_samples,
        )

        return source_access, target_access

    def _create_digits_access(self):
        (
            source_access,
            target_access,
            self._num_channels,
        ) = digits.DigitDataset.get_accesses(
            digits.DigitDataset(self._data_config["source"].upper()),
            digits.DigitDataset(self._data_config["target"].upper()),
            data_path=self._data_path,
        )
        return source_access, target_access

    def _create_office31_access(self):
        source_access, target_access = office.Office31Dataset.get_accesses(
            office.Office31Dataset(self._data_config["source"].lower()),
            office.Office31Dataset(self._data_config["target"].lower()),
            data_path=self._data_path,
        )
        return source_access, target_access
