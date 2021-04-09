import numpy as np
import scipy.stats as ss
import os
import logging

from sklearn.utils import check_random_state

import torch
from torch.utils.data import Dataset
import ada.utils.experimentation as xp
from ada.datasets.dataset_access import DatasetAccess


def shift_data(x_in, ti=None, ri=None, si=None):
    """
    This function applies scaling, translation and/or rotation to 2D data points, in that order only.

    Args
        x_in (np.ndarray): data, input feature array of shape (n, d)
        ti (float, optional): translation (scalar or vector of compatible dimension). Defaults to None.
        ri (float, optional): rotation angle in radians (scalar, for 2D points only). Defaults to None.
        si (float, optional): scaling factor (scalar). Defaults to None.

    Returns:
        np.ndarray: transformed feature array of shape (n, d), same as x_in.
    """
    x_out = x_in

    if si is not None and si > 0:
        s_mat = si * np.eye(x_in.shape[1])
        x_out = x_out @ s_mat

    if ti is not None:
        x_out = x_out + ti

    if ri is not None:
        if x_in.shape[1] != 2:
            raise ValueError("Rotation may be applied to 2D data only")
        rot_mat = np.array([[np.cos(ri), np.sin(ri)], [-np.sin(ri), np.cos(ri)]])
        x_out = x_out @ rot_mat

    return x_out


def gen_cluster_distributions(
    dim, n_clusters, radius, random_state=None, centers="normal"
):
    random_state = check_random_state(random_state)
    if isinstance(centers, list):
        centers = np.array(centers)
    if isinstance(centers, str):
        if centers == "normal":
            cluster_means = random_state.normal(size=(n_clusters, dim))
        elif centers == "fixed" and n_clusters < 3 and dim == 2:
            fixed_means = np.array([[-0.5, 0.0], [0.5, 0]])
            cluster_means = fixed_means[:n_clusters, :]
    elif isinstance(centers, np.ndarray):
        cluster_means = centers
        n_clusters, dim = cluster_means.shape
    else:
        cluster_means = random_state.uniform(size=(n_clusters, dim))
    # cluster_std = random_state.uniform(size=(n_clusters, dim)) * radius
    if isinstance(radius, (np.ndarray, list)):
        radius = np.array(radius)
        if radius.shape != (n_clusters, dim):
            logging.debug(radius.shape, centers.shape)
            n_radii, dim_radius = (
                radius.shape if radius.ndim == 2 else radius.shape[0],
                1,
            )
            if dim_radius != dim and radius.ndim > 1 and n_radii == n_clusters:
                cluster_var = np.repeat(radius[:, 0], dim).reshape((n_clusters, dim))
            elif dim_radius != dim and radius.ndim == 1 and n_radii == n_clusters:
                cluster_var = np.repeat(radius, dim).reshape((n_clusters, dim))
            elif dim_radius == dim and n_radii == 1:
                cluster_var = (
                    np.repeat(radius[:], n_clusters).reshape((dim, n_clusters)).T
                )
            else:
                cluster_var = np.repeat(radius[0], dim * n_clusters).reshape(
                    (dim, n_clusters)
                )
            logging.warning(
                f"Input radius {radius} shape doesn't match cluster centers shape. Attempts to adapt, will use {cluster_var} instead"
            )
        else:
            cluster_var = radius
    else:
        cluster_var = np.ones((n_clusters, dim)) * radius
    if n_clusters <= 1:
        cluster_dist = ss.multivariate_normal(
            mean=cluster_means.flatten(), cov=cluster_var.flatten()
        )
        return cluster_dist, cluster_means, cluster_var

    cluster_dists = np.array(
        list(
            map(
                lambda x: (ss.multivariate_normal, {"mean": x[0], "cov": x[1]}),
                zip(cluster_means, cluster_var),
            )
        )
    )
    return cluster_dists, cluster_means, cluster_var


class CausalClusterGenerator:
    """
    Generate blobs from a gaussian distribution following given causal parameters relating environment/domain, X and Y:
    - Y --> X: select class Y, then distribution X|Y
    """

    def __init__(
        self,
        dim=2,
        n_clusters=2,
        radius=0.05,
        proba_classes=0.5,
        centers="fixed",
        shape="blobs",
        data_seed=None,
    ):
        self._random_state = check_random_state(data_seed)
        self._n_clusters = n_clusters
        self._proba_classes = proba_classes
        self.shape = shape
        self._cluster_dists, self._means, self._stds = gen_cluster_distributions(
            dim=dim,
            n_clusters=n_clusters,
            radius=radius,
            centers=centers,
            random_state=self._random_state,
        )

    def generate_sample(
        self,
        nb_samples,
        shift_y=False,
        shift_x=False,
        shift_conditional_x=False,
        shift_conditional_y=False,
        y_cause_x=True,
        ye=0.5,
        te=0.3,
        se=None,
        re=None,
    ):
        """
        Generate a sample and apply a given shift:
        shift_x = change p(x), ie x_e = f(x, env)
        shift_y = change p(y), ie y_e = f(y, env)
        shift_conditional_x = change p(x|y), ie x_e = f(y, x, env)
        shift_conditional_y = change p(y|x), ie y_e = f(x, y, env)

        env_parameters control the change in the data:
        ye = proportion of class 0 labels
        te = translation value (uniform on all dimensions!)
        se = scaling factor
        re = rotation in radians
        """
        if shift_y and y_cause_x:
            logging.debug("E --> Z=Y")
            zy = ss.bernoulli(ye * self._proba_classes).rvs(
                size=nb_samples, random_state=self._random_state
            )
            zx = None
        elif (
            isinstance(self._proba_classes, (np.ndarray, list))
            or len(self._cluster_dists) > 2
        ):
            n_clusters, dim = self._means.shape
            if not isinstance(self._proba_classes, (np.ndarray, list)):
                n_samples = (np.ones(n_clusters, dtype=float) / n_clusters) * nb_samples
            else:
                probas = np.array(self._proba_classes)
                probas /= probas.sum()
                n_samples = probas * nb_samples
            n_samples = n_samples.astype(np.int)
            n_samples[-1] = nb_samples - np.sum(n_samples[:-1])
            zy = np.empty(nb_samples, dtype=np.int)
            zx = np.empty((nb_samples, dim), dtype=np.float)
            sid = 0
            for class_id, n_class_samples in enumerate(n_samples):
                pdist, law_args = self._cluster_dists[class_id]
                zy[sid : sid + n_class_samples] = np.ones(n_class_samples) * class_id
                zx[sid : sid + n_class_samples, :] = pdist.rvs(
                    size=n_class_samples, random_state=self._random_state, **law_args
                )
                sid += n_class_samples
        else:
            logging.debug("ZY = cte")
            zy = ss.bernoulli(self._proba_classes).rvs(
                size=nb_samples, random_state=self._random_state
            )
            zx = None

        logging.debug("ZY --> ZX(ZY)")
        if zx is None:
            zx = np.array(
                [
                    pdist.rvs(size=1, random_state=self._random_state, **law_args)
                    for pdist, law_args in self._cluster_dists[zy]
                ]
            ).astype(np.float32)

        if self.shape.lower() == "moons":
            r = 1 - zy * 2  # assumes 2 classes, maps 0 to 1 and 1 to -1
            indices = np.linspace(0, np.pi, nb_samples)
            self._random_state.shuffle(indices)
            zx[:, 0] = zx[:, 0] + r * np.cos(indices)
            zx[:, 1] = zx[:, 1] + r * np.sin(indices)

        if shift_x:
            logging.debug("E, ZX --> X = g_E(XZ)")
            x = shift_data(zx, ti=te, si=se, ri=re)
        else:
            logging.debug("X=ZX")
            x = zx

        if shift_conditional_x:
            logging.debug("ZY, ZX, E --> g_E(X, Y)")
            # x = f(y, env)
            if te is None:
                ti0 = ti1 = None
            elif isinstance(te, float):
                ti0, ti1 = te * 2, te / 2
            else:
                ti0, ti1 = te
            if se is None:
                si0 = si1 = se
            elif isinstance(se, float):
                si0, si1 = se * 2, se / 2
            else:
                si0, si1 = se
            if se is not None and (si0 < 0 or si1 < 0):
                raise ValueError("Scaling factor cannot be negative")
            if re is None:
                ri0 = ri1 = re
            elif isinstance(re, float):
                ri0, ri1 = re * 2, re / 2
            else:
                ri0, ri1 = re
            x[zy == 0, :] = shift_data(zx[zy == 0], ti=ti0, si=si0, ri=ri0)
            x[zy == 1, :] = shift_data(zx[zy == 1], ti=ti1, si=si1, ri=ri1)

        if y_cause_x:
            logging.debug("Y=ZY")
            y = zy
            return x, y

        fx = np.sum(x, axis=1)
        xm = self._means.sum(axis=1)
        if shift_conditional_y:
            logging.debug("X, E --> Y")
            # y = f(env, x)
            thresh = np.percentile(xm, q=ye * 100)
        else:
            # y = f(x) indep. env
            logging.debug("E --> X --> Y")
            thresh = np.median(xm)

        logging.debug("threshold:", thresh)
        y = (fx > thresh).astype(int)
        if shift_y:
            logging.debug("flip random labels")
            idx = np.random.choice(len(y), int(ye * len(y)), replace=False)
            y[idx] = 1

        return x, y

    @property
    def means(self):
        return self._means


def get_datashift_params(data_shift=None, ye=0.5, te=None, se=None, re=None):
    """
    This factory simplifies the parameter generation process for a number of
    use cases. The parameters generated can be used with CausalClusterGenerator.generate_sample
    """
    data_shift_types = dict(
        no_shift=dict(
            shift_y=False,
            shift_x=False,
            shift_conditional_x=False,
            shift_conditional_y=False,
            y_cause_x=True,
            ye=ye,
            te=te,
            se=se,
            re=re,
        ),
        covariate_shift_y=dict(
            y_cause_x=True, shift_y=False, shift_x=True, re=re, te=te, se=se
        ),
        cond_covariate_shift_y=dict(
            y_cause_x=True,
            shift_y=False,
            shift_conditional_x=True,
            shift_x=False,
            re=re,
            te=te,
            se=se,
        ),
        covariate_shift_x=dict(
            y_cause_x=True, shift_y=False, shift_x=True, re=re, te=te, se=se
        ),
        label_shift=dict(y_cause_x=True, shift_y=True, shift_x=False, ye=ye),
        label_and_covariate_shift=dict(
            y_cause_x=True, shift_y=True, shift_x=True, ye=ye, re=re, te=te, se=se
        ),
        label_and_cond_covariate_shift=dict(
            y_cause_x=True,
            shift_y=True,
            shift_conditional_x=True,
            ye=ye,
            re=re,
            te=te,
            se=se,
        ),
        covariate_and_cond_label_shift=dict(
            y_cause_x=False,
            shift_x=True,
            shift_conditional_y=True,
            ye=ye,
            re=re,
            te=te,
            se=se,
        ),
    )

    if data_shift is not None:
        return data_shift_types[data_shift]

    return list(data_shift_types.keys())


class CausalBlobs(torch.utils.data.Dataset):
    """
    `CausalGaussianBlobs Dataset.
    MNIST-like dataset that generates Blobs in a given environment setting
    - original cluster params set by `cluster_params` dictionary
    - environment and cluster generation params given by `transform` dictionary
    """

    raw_folder = "BlobsData"

    def __init__(
        self,
        data_path,  # for compatibility with other datasets API
        train=True,
        transform=None,
        download=True,
        cluster_params=None,
        n_samples=300,
    ):
        """Init Blobs dataset."""
        super(CausalBlobs, self).__init__()
        self.root = data_path
        self.transform = transform if transform is not None else {}
        self.train = train  # training set or test set
        self.n_samples = n_samples

        if cluster_params is None:
            self.cluster_params = dict(
                n_clusters=2, data_seed=0, radius=0.02, centers=None, proba_classes=0.5
            )
        else:
            self.cluster_params = cluster_params

        tmp_cluster_params = cluster_params.copy()
        if isinstance(cluster_params["centers"], np.ndarray):
            tmp_cluster_params["centers"] = tmp_cluster_params["centers"].tolist()
        cluster_hash = xp.param_to_hash(tmp_cluster_params)
        transform_hash = xp.param_to_hash(transform)
        self.data_dir = os.path.join(cluster_hash, transform_hash)
        root_dir = os.path.join(self.root, self.raw_folder)
        os.makedirs(root_dir, exist_ok=True)
        xp.record_hashes(
            os.path.join(root_dir, "parameters.json"),
            f"{cluster_hash}/{transform_hash}",
            {"cluster_params": tmp_cluster_params, "transform": transform},
        )

        self.training_file = "causal_blobs_train.pt"
        self.test_file = "causal_blobs_test.pt"
        self._cluster_gen = None

        if not self._check_exists() or download:
            self.create_on_disk()

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        if self.train:
            self.data, self.targets = torch.load(
                os.path.join(
                    self.root, self.raw_folder, self.data_dir, self.training_file
                )
            )
        else:
            self.data, self.targets = torch.load(
                os.path.join(self.root, self.raw_folder, self.data_dir, self.test_file)
            )

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = self.data[index], self.targets[index]

        return data, target

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(
            os.path.join(self.root, self.raw_folder, self.data_dir, self.training_file)
        ) and os.path.exists(
            os.path.join(self.root, self.raw_folder, self.data_dir, self.test_file)
        )

    def create_on_disk(self):
        file_path = os.path.join(self.root, self.raw_folder, self.data_dir)
        # make data dirs
        os.makedirs(file_path, exist_ok=True)

        self._cluster_gen = CausalClusterGenerator(**self.cluster_params)

        X_tr, y_tr = self._cluster_gen.generate_sample(self.n_samples, **self.transform)

        Xtr = torch.from_numpy(X_tr).float()
        ytr = torch.from_numpy(y_tr).long()
        training_set = (Xtr, ytr)

        X_te, y_te = self._cluster_gen.generate_sample(self.n_samples, **self.transform)
        Xte = torch.from_numpy(X_te).float()
        yte = torch.from_numpy(y_te).long()
        test_set = (Xte, yte)

        with open(os.path.join(file_path, self.training_file), "wb") as f:
            torch.save(training_set, f)
        with open(os.path.join(file_path, self.test_file), "wb") as f:
            torch.save(test_set, f)

    def delete_from_disk(self):
        file_path = os.path.join(self.root, self.raw_folder, self.data_dir)
        os.remove(os.path.join(file_path, self.training_file))
        os.remove(os.path.join(file_path, self.test_file))


class CausalBlobsDataAccess(DatasetAccess):
    def __init__(self, data_path, transform, download, cluster_params, n_samples):
        super().__init__(n_classes=cluster_params.get("n_clusters", 2))
        self._data_path = data_path
        self._transform = transform
        self._download = download
        self._cluster_params = cluster_params
        self._n_samples = n_samples

    def get_train(self):
        return CausalBlobs(
            data_path=self._data_path,
            train=True,
            transform=self._transform,
            download=self._download,
            cluster_params=self._cluster_params,
            n_samples=self._n_samples,
        )

    def get_test(self):
        return CausalBlobs(
            data_path=self._data_path,
            train=False,
            transform=self._transform,
            download=self._download,
            cluster_params=self._cluster_params,
            n_samples=self._n_samples,
        )
