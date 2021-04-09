import os
import shutil
import logging
import random
import json
import hashlib
import glob
import re
import torch
from copy import deepcopy
from datetime import datetime
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import LoggerCollection
from pytorch_lightning.callbacks import ModelCheckpoint

from ada.models.network_factory import NetworkFactory
import ada.models.architectures as archis


def create_timestamp_string(fmt="%Y-%m-%d.%H.%M.%S.%f"):
    now = datetime.now()
    time_str = now.strftime(fmt)
    return time_str


def load_json_dict(conf_filename):
    with open(conf_filename, "r") as conf_file:
        conf = json.load(conf_file)
    return conf


def set_all_seeds(seed):
    """See https://pytorch.org/docs/stable/notes/randomness.html

    We activate the PyTorch options for best reproducibility. Note that this may be detrimental
    to processing speed, as per the above documentation:

      ...the processing speed (e.g. the number of batches trained per second) may be lower
      than when the model functions nondeterministically.
      However, even though single-run speed may be slower, depending on your application
      determinism may save time by facilitating experimentation, debugging,
      and regression testing.

    Args:
        seed (int): the seed which will be used for all random generators.
    """
    random.seed(seed)

    # pytorch RNGs
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # numpy RNG
    np.random.seed(seed)


def param_to_str(param_dict):
    def key_val_mapper(kv):
        if isinstance(kv[1], dict):
            return param_to_str(kv[1])
        if isinstance(kv[1], float):
            return f"{kv[0]}{kv[1]:.2f}"
        if isinstance(kv[1], bool):
            return kv[0] if kv[1] else f"no-{kv[0]}"
        if isinstance(kv[1], str):
            return kv[1]
        if isinstance(kv[1], np.ndarray):
            # return "array"
            return "x".join(map(str, kv[1].flatten()))
        return f"{kv[0]}{kv[1]}"

    return "-".join(map(key_val_mapper, param_dict.items()))


def param_to_hash(param_dict):
    config_hash = hashlib.md5(
        json.dumps(param_dict, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return config_hash


def record_hashes(hash_file, hash_, value):
    if os.path.exists(hash_file):
        with open(hash_file, "r") as fd:
            known_hashes = json.load(fd)
    else:
        known_hashes = {}

    if hash_ not in known_hashes:
        known_hashes[hash_] = value
        with open(hash_file, "w") as fd:
            json.dump(known_hashes, fd)
        return True
    return False


def train_model(
    method,
    data_factory,
    train_params=None,
    archi_params=None,
    method_name=None,
    method_params=None,
    seed=98347,
    fix_few_seed=0,
    gpus=None,
    mlflow_uri=None,
    tensorboard_dir=None,
    checkpoint_dir=None,
    fast=False,
    try_to_resume=True,
):
    """This is the main function where a single model is created and trained, for a single seed value.

    Args:
        method (archis.Method): type of method, used to decide which networks to build and
            how to use some parameters.
        data_factory (DataFactory): dataset description to get dataset loaders, as well as useful
            information for some networks.
        train_params (dict, optional): Hyperparameters for training (see network config). Defaults to None.
        archi_params (dict, optional): Parameters of the network (see network config). Defaults to None.
        method_name (string, optional): A unique name describing the method, with its parameters. Used for logging results.
            Defaults to None.
        method_params (dict, optional): Parameters to be fed to the model that are specific to `method`. Defaults to None.
        seed (int, optional): Global seed for reproducibility. Defaults to 98347.
        fix_few_seed (int, optional): See for semi-supervised setting, fixing which target samples are labeled. Defaults to 0.
        gpus (list of int, optional): Which GPU ids to use. Defaults to None.
        mlflow_uri (int|string, optional): if a string, must be formatted like <uri>:<port>. If a port, will try
            to log to a MLFlow server on localhost:port. If None, ignores MLFlow logging. Defaults to None.
        fast (bool, optional): Whether to activate the `fast_dev_run` option of PyTorch-Lightning,
            training only on 1 batch per epoch for debugging. Defaults to False.

    Returns:
        2-elements tuple containing:

            - pl.Trainer: object containing the resulting metrics, used for evaluation.
            - BaseAdaptTrainer: pl.LightningModule object (derived class depending on `method`), containing
                both the dataset & trained networks.

    """
    if type(method) is str:
        method = archis.Method(method)
    if method_name is None:
        method_name = method.value
    train_params_local = deepcopy(train_params)

    set_all_seeds(seed)
    if fix_few_seed > 0:
        archi_params["random_state"] = fix_few_seed
    else:
        archi_params["random_state"] = seed

    dataset = data_factory.get_multi_domain_dataset(seed)
    n_classes, data_dim, args = data_factory.get_data_args()
    network_factory = NetworkFactory(archi_params)
    # setup feature extractor
    feature_network = network_factory.get_feature_extractor(data_dim, *args)
    # setup classifier
    feature_dim = feature_network.output_size()
    classifier_network = network_factory.get_task_classifier(feature_dim, n_classes)

    method_params = {} if method_params is None else method_params
    if method.is_mmd_method():
        model = archis.create_mmd_based(
            method=method,
            dataset=dataset,
            feature_extractor=feature_network,
            task_classifier=classifier_network,
            **method_params,
            **train_params_local,
        )
    else:
        critic_input_size = feature_dim
        # setup critic network
        if method.is_cdan_method():
            if method_params is not None and method_params.get("use_random", False):
                critic_input_size = method_params["random_dim"]
            else:
                critic_input_size = feature_dim * n_classes
        critic_network = network_factory.get_critic_network(critic_input_size)

        model = archis.create_dann_like(
            method=method,
            dataset=dataset,
            feature_extractor=feature_network,
            task_classifier=classifier_network,
            critic=critic_network,
            **method_params,
            **train_params_local,
        )

    data_name = data_factory.get_data_short_name()

    if checkpoint_dir is not None:
        path_method_name = re.sub(r"[^-/\w\.]", "_", method_name)
        full_checkpoint_dir = os.path.join(
            checkpoint_dir, path_method_name, f"seed_{seed}"
        )
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(full_checkpoint_dir, "{epoch}"),
            monitor="last_epoch",
            mode="max",
        )
        checkpoints = sorted(
            glob.glob(f"{full_checkpoint_dir}/*.ckpt"), key=os.path.getmtime
        )
        if len(checkpoints) > 0 and try_to_resume:
            last_checkpoint_file = checkpoints[-1]
            if method is archis.Method.WDGRL:
                # WDGRL doesn't resume training gracefully
                last_epoch = (
                    train_params_local["nb_init_epochs"]
                    + train_params_local["nb_adapt_epochs"]
                )
                if f"epoch={last_epoch - 1}" not in last_checkpoint_file:
                    last_checkpoint_file = None
        else:
            last_checkpoint_file = None
    else:
        checkpoint_callback = None
        last_checkpoint_file = None

    if mlflow_uri is not None:
        if mlflow_uri.isdecimal():
            mlflow_uri = f"http://127.0.0.1:{mlflow_uri}"
        mlf_logger = MLFlowLogger(
            experiment_name=data_name,
            tracking_uri=mlflow_uri,
            tags=dict(
                method=method_name,
                data_variant=data_factory.get_data_long_name(),
                script=__file__,
            ),
        )
    else:
        mlf_logger = None

    if tensorboard_dir is not None:
        tnb_logger = TensorBoardLogger(
            save_dir=tensorboard_dir,
            name=f"{data_name}_{method_name}",
        )
    else:
        tnb_logger = None

    loggers = [logger for logger in [mlf_logger, tnb_logger] if logger is not None]
    if len(loggers) == 0:
        logger = False
    else:
        logger = LoggerCollection(loggers)
        logger.log_hyperparams(
            {
                "seed": seed,
                "feature_network": archi_params["feature"]["name"],
                "method group": method.value,
                "method": method_name,
                "start time": create_timestamp_string("%Y-%m-%d %H:%M:%S"),
            }
        )

    max_nb_epochs = (
        train_params_local["nb_adapt_epochs"] * 5
        if method is archis.Method.WDGRLMod
        else train_params["nb_adapt_epochs"]
    )
    pb_refresh = 1 if len(dataset) < 1000 else 10
    row_log_interval = max(10, len(dataset) // train_params_local["batch_size"] // 10)

    if gpus is not None and len(gpus) > 1 and method is archis.Method.WDGRL:
        logging.warning("WDGRL is not compatible with multi-GPU.")
        gpus = [gpus[0]]

    trainer = pl.Trainer(
        progress_bar_refresh_rate=pb_refresh,  # in steps
        row_log_interval=row_log_interval,
        min_epochs=train_params_local["nb_init_epochs"],
        max_epochs=max_nb_epochs + train_params_local["nb_init_epochs"],
        early_stop_callback=False,
        num_sanity_val_steps=5,
        check_val_every_n_epoch=1,
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=last_checkpoint_file,
        gpus=gpus,
        logger=logger,
        weights_summary=None,  # 'full' is default
        fast_dev_run=fast,
    )

    if last_checkpoint_file is None:
        logging.info(f"Training model with {method.name} {param_to_str(method_params)}")
    else:
        logging.info(
            f"Resuming training with {method.name} {param_to_str(method_params)}, from {last_checkpoint_file}."
        )
    trainer.fit(model)
    if trainer.interrupted:
        raise KeyboardInterrupt("Trainer was interrupted and shutdown gracefully.")

    if logger:
        logger.log_hyperparams(
            {"finish time": create_timestamp_string("%Y-%m-%d %H:%M:%S")}
        )
    return trainer, model


def loop_train_test_model(
    method,
    results,
    nseeds,
    backup_file,
    test_params,
    data_factory,
    gpus,
    force_run=False,
    progress_callback=lambda percent: None,
    method_name=None,
    method_params=None,
    mlflow_uri=None,
    tensorboard_dir=None,
    checkpoint_dir=None,
):
    init_seed = 34875
    seeds = np.random.RandomState(init_seed).randint(100, 100000, size=nseeds)
    if type(method) is str:
        method = archis.Method(method)
    if method_name is None:
        method_name = method.value
    if method_params is None:
        method_params = {}

    if data_factory.is_semi_supervised() and not method.is_fewshot_method():
        logging.warning(
            f"Skipping {method_name}: not suited for the semi-supervised setting."
        )
        return None

    res_archis = {}
    for i, seed in enumerate(tqdm(seeds)):
        if results.already_computed(method_name, seed) and not force_run:
            progress_callback((i + 1) / nseeds)
            continue

        trainee, trained_archi = train_model(
            method,
            seed=seed,
            data_factory=data_factory,
            gpus=gpus,
            method_name=method_name,
            method_params=method_params,
            mlflow_uri=mlflow_uri,
            tensorboard_dir=tensorboard_dir,
            checkpoint_dir=checkpoint_dir,
            try_to_resume=not force_run,
            **test_params,
        )
        # validation scores
        results.update(
            is_validation=True,
            method_name=method_name,
            seed=seed,
            metric_values=trainee.callback_metrics,
        )
        # test scores
        trainee.test()
        results.update(
            is_validation=False,
            method_name=method_name,
            seed=seed,
            metric_values=trainee.callback_metrics,
        )
        results.to_csv(backup_file)
        results.print_scores(
            method_name,
            stdout=True,
            fdout=None,
            print_func=tqdm.write,
        )
        res_archis[seed] = trained_archi
        progress_callback((i + 1) / nseeds)

    best_archi_seed = results.get_best_archi_seed()
    if best_archi_seed not in res_archis:
        return None
    return res_archis[best_archi_seed]
