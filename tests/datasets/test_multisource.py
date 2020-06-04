import unittest
import numpy as np

from ada.utils.experimentation import set_all_seeds
from ada.datasets.multisource import MultiDomainDatasets, DatasetSizeType
from ada.datasets.toys import CausalBlobsDataAccess
from ada.datasets.sampler import SamplingConfig


class TestMultiDomainDatasets(unittest.TestCase):
    def setUp(self):
        n_samples = 300
        cluster_params = dict(
            dim=2,
            n_clusters=2,
            radius=0.05,
            proba_classes=0.3,
            centers=np.array([[-0.5, 0.5], [0.5, -0.5]]),
            data_seed=123,
        )

        no_shift_args = dict(
            shift_y=False,
            shift_x=False,
            shift_conditional_x=False,
            shift_conditional_y=False,
            y_cause_x=True,
            ye=0.5,
            te=0.3,
            se=None,
            re=None,
        )

        shift_params = dict(
            y_cause_x=True, shift_y=False, shift_x=True, re=np.pi / 2, te=0
        )

        self.source_access = CausalBlobsDataAccess(
            data_path="blobs",
            transform=no_shift_args,
            download=True,
            cluster_params=cluster_params,
            n_samples=n_samples,
        )

        self.target_access = CausalBlobsDataAccess(
            data_path="blobs",
            transform=shift_params,
            download=True,
            cluster_params=cluster_params,
            n_samples=n_samples // 2,
        )

    def test_batch_natural(self):
        set_all_seeds(123)
        domain_datasets = MultiDomainDatasets(
            source_access=self.source_access,
            target_access=self.target_access,
            size_type=DatasetSizeType.Source,
            n_fewshot=0,
        )
        domain_datasets.prepare_data_loaders()

        batch_size = 32
        mdloader = domain_datasets.get_domain_loaders(
            split="train", batch_size=batch_size
        )

        n_train_samples = 270
        self.assertEqual(
            len(mdloader._dataloaders[0].batch_sampler.sampler), n_train_samples
        )
        n_batches = n_train_samples // batch_size
        self.assertEqual(len(mdloader), n_batches)
        for e in range(100):
            for i, batch in enumerate(mdloader):
                self.assertEqual(len(batch), 2)
                self.assertEqual(len(batch[0]), 2)
                self.assertEqual(len(batch[1]), 2)
                self.assertEqual(len(batch[0][0]), batch_size)
                self.assertEqual(len(batch[1][0]), batch_size)

    def test_batch_balance(self):
        set_all_seeds(123)
        domain_datasets = MultiDomainDatasets(
            source_access=self.source_access,
            target_access=self.target_access,
            source_sampling_config=SamplingConfig(balance=True),
            target_sampling_config=SamplingConfig(balance=True),
            size_type=DatasetSizeType.Source,
            n_fewshot=0,
        )
        domain_datasets.prepare_data_loaders()

        batch_size = 32
        mdloader = domain_datasets.get_domain_loaders(
            split="train", batch_size=batch_size
        )

        n_train_samples = 270
        self.assertEqual(
            mdloader._dataloaders[0].batch_sampler.n_dataset, n_train_samples
        )
        n_batches = n_train_samples // batch_size
        self.assertEqual(len(mdloader), n_batches)
        for e in range(100):
            for i, batch in enumerate(mdloader):
                self.assertEqual(len(batch), 2)
                self.assertEqual(len(batch[0]), 2)
                self.assertEqual(len(batch[1]), 2)
                self.assertEqual(batch[0][1].sum(), (1 - batch[0][1]).sum())
                self.assertEqual(batch[1][1].sum(), (1 - batch[1][1]).sum())
                self.assertEqual(len(batch[0][0]), batch_size)
                self.assertEqual(len(batch[1][0]), batch_size)

    def test_batch_natural_few(self):
        set_all_seeds(123)
        domain_datasets = MultiDomainDatasets(
            source_access=self.source_access,
            target_access=self.target_access,
            size_type=DatasetSizeType.Source,
            n_fewshot=1,
        )
        domain_datasets.prepare_data_loaders()

        batch_size = 32
        mdloader = domain_datasets.get_domain_loaders(
            split="train", batch_size=batch_size
        )

        n_train_samples = 270
        self.assertEqual(
            len(mdloader._dataloaders[0].batch_sampler.sampler), n_train_samples
        )
        n_batches = n_train_samples // batch_size
        self.assertEqual(len(mdloader), n_batches)
        for e in range(100):
            for i, batch in enumerate(mdloader):
                self.assertEqual(len(batch), 3)
                self.assertEqual(len(batch[0]), 2)
                self.assertEqual(len(batch[1]), 2)
                self.assertEqual(len(batch[0][0]), batch_size)
                self.assertEqual(len(batch[2][0]), batch_size)
                self.assertEqual(len(batch[1][0]), 2)
                self.assertEqual(len(batch[1][1]), 2)
