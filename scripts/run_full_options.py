"""
This script demonstrates the full set of options available
from the library.
- select from a number of datasets
- use MLFlow to visualize the loss during training
- switch from unsupervised to supervised learning with a few labeled samples
"""
import os
from pathlib import Path
import logging
import argparse

from ada.datasets.dataset_factory import DatasetFactory, WeightingType
import ada.utils.experimentation as xp
import ada.utils.experimentation_results as xpr
import ada.models.architectures as archis

from ada.utils.plotting import plot_archi_data
import matplotlib.pyplot as plt


def get_best_archis_for_seed(
    seed,
    test_params,
    data_factory,
    gpus,
    methods_variant_params,
    mlflow_uri,
    tensorboard_dir,
    checkpoint_dir,
):
    best_archis = {}
    for method_name, method_params in sorted(methods_variant_params.items()):
        method = archis.Method(method_params["method"])
        method_params = method_params.copy()
        del method_params["method"]

        if data_factory.is_semi_supervised() and not method.is_fewshot_method():
            logging.warning(
                f"Skipping {method_name}: not suited for the semi-supervised setting."
            )
            continue

        _, trained_archi = xp.train_model(
            method,
            seed=seed,
            data_factory=data_factory,
            gpus=gpus,
            method_params=method_params,
            method_name=method_name,
            mlflow_uri=mlflow_uri,
            tensorboard_dir=tensorboard_dir,
            checkpoint_dir=checkpoint_dir,
            try_to_resume=True,
            **test_params,
        )
        best_archis[method_name] = trained_archi
    return best_archis


if __name__ == "__main__":
    format_str = "%(asctime)s %(name)s [%(levelname)s] - %(message)s"
    logging.basicConfig(format=format_str)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    default_toy_data_file = "../configs/datasets/toy_2_moons.json"

    parser = argparse.ArgumentParser(
        description="Train & evaluate domain adaptation architecture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--dataconf",
        help="path to config file defining dataset parameters.",
        type=str,
        default=default_toy_data_file,
    )
    parser.add_argument(
        "-m",
        "--method",
        help="path to config file defining which methods to use and their parameters.",
        type=str,
        default="../configs/default_methods.json",
    )
    parser.add_argument(
        "--nseeds",
        help="Number of seeds to run with. Each method will be run `nseeds` times with the same set of seeds."
        "The seeding controls: the train/test split ; the few-shot samples selection ; "
        "the network initialization ; batch sampling.",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--outdir",
        help="output dir",
        type=str,
        default=os.path.join(str(Path.home()), "sync", "ada"),
    )
    parser.add_argument(
        "--data_path",
        help="Full path to input data directory",
        type=str,
        default=os.path.join(str(Path.home()), "sync", "data"),
    )
    parser.add_argument(
        "-f",
        "--fewshot",
        help="number of labeled items in target domain",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--force",
        help="Force run even if algorithm has been run already",
        action="store_true",
    )
    parser.add_argument("-g", "--gpu", nargs="*", help="id of gpu to use", type=int)
    parser.add_argument(
        "--mlflow",
        help="MLFlow UI server port on localhost, or full URI (http(s):, file:). None to deactivate.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--tensorboard_dir",
        help="Directory for tensorboard files. None to deactivate.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-n",
        "--netconf",
        help="path to config file defining network parameters and learning hyperparameters.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-w",
        "--weighted",
        help="reweight dataset from preset values",
        type=str,
        default=None,
        choices=[w.value for w in WeightingType],
    )
    parser.add_argument(
        "-s",
        "--fast",
        help="limit training data to single training batch for fast checking",
        action="store_true",
    )

    args = parser.parse_args()
    logging.info(f"Running with {args}")
    nseeds = args.nseeds
    data_params = xp.load_json_dict(args.dataconf)

    if args.weighted is not None:
        data_params["weight_type"] = args.weighted

    network_config_file = args.netconf
    if network_config_file is None:
        data_name = data_params["dataset_group"]
        network_config_file = f"../configs/{data_name}_network.json"
    if not os.path.exists(network_config_file):
        raise ValueError(
            f"{network_config_file} doesn't exist.\
            You need to create it or specify a `netconf` parameter with a valid config file."
        )
    network_params = xp.load_json_dict(network_config_file)

    test_params = network_params.copy()

    if args.fewshot > 0:
        test_params["fix_few_seed"] = 8751354

    test_params["fast"] = args.fast

    data_factory = DatasetFactory(
        data_params, data_path=args.data_path, n_fewshot=args.fewshot
    )
    output_dir = os.path.join(args.outdir, data_factory.get_data_short_name())

    os.makedirs(output_dir, exist_ok=True)

    # parameters that change across experiments for the same dataset
    record_params = test_params.copy()
    record_params.update(
        {
            k: v
            for k, v in data_params.items()
            if k not in ("dataset_group", "dataset_name")
        }
    )
    params_hash = xp.param_to_hash(record_params)
    hash_file = os.path.join(output_dir, "parameters.json")
    xp.record_hashes(hash_file, params_hash, record_params)
    output_file_prefix = os.path.join(output_dir, params_hash)
    test_csv_file = f"{output_file_prefix}.csv"
    checkpoint_dir = os.path.join(output_dir, "checkpoints", params_hash)

    methods_variant_params = xp.load_json_dict(args.method)

    results = xpr.XpResults.from_file(["source acc", "target acc"], test_csv_file)
    if len(results) == 0 or args.force:
        if args.force:
            results.remove(method_names=list(methods_variant_params))
            results.to_csv(test_csv_file)

    do_plots = False

    archis_res = {}
    for method_name, method_params in sorted(methods_variant_params.items()):
        method = archis.Method(method_params["method"])
        method_params = method_params.copy()
        del method_params["method"]
        domain_archi = xp.loop_train_test_model(
            method,
            results,
            nseeds,
            test_csv_file,
            test_params=test_params,
            data_factory=data_factory,
            gpus=args.gpu,
            force_run=args.force,
            method_name=method_name,
            method_params=method_params,
            mlflow_uri=args.mlflow,
            tensorboard_dir=args.tensorboard_dir,
            checkpoint_dir=checkpoint_dir,
        )
        archis_res[method_name] = domain_archi
        if domain_archi is not None:
            # plot only if at least one model changed
            do_plots = True
        results.to_csv(test_csv_file)

    print(results.get_data())
    logging.info(
        "Recomputing context for unique seed with best average results over all methods"
    )
    mean_seed = results.get_mean_seed("target acc")
    archis_res = get_best_archis_for_seed(
        seed=mean_seed,
        test_params=test_params,
        data_factory=data_factory,
        gpus=args.gpu,
        methods_variant_params=methods_variant_params,
        mlflow_uri=args.mlflow,
        tensorboard_dir=args.tensorboard_dir,
        checkpoint_dir=checkpoint_dir,
    )
    output_file_prefix = "_".join((output_file_prefix, str(mean_seed)))

    results.append_to_markdown(
        filepath=os.path.join(output_dir, "all_res.md"),
        test_params=test_params,
        nseeds=nseeds,
        splits=["Test"],
    )

    if do_plots:
        logging.getLogger("matplotlib").setLevel(logging.ERROR)

        for name, archi in archis_res.items():
            if archi is None:
                logging.warning(f"Cannot plot for {name}.")
                continue
            plot_archi_data(
                archi,
                name,
                save_prefix=output_file_prefix,
                plot_f_lines=False,
                plot_features=["PCA", "TSNE", "UMAP"],
                do_domain_boundary=True,
                do_entropy=False,
            )
            plt.close("all")

    logging.info(f"See results with prefix {output_file_prefix}")
