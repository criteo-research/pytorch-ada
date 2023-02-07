"""
This script gives an example use of the library for
classical unsupervised domain adaptation experiments.

For a more complex example including few-shot, semi-supervised 
or generalized target shift domain adaptation,
please refer to the `run_full_options.py` script.
"""
import os
from pathlib import Path
import logging
import argparse

from ada.datasets.dataset_factory import DatasetFactory
import ada.utils.experimentation as xp
import ada.models.architectures as archis
import ada.utils.experimentation_results as xpr
from ada.utils.plotting import plot_archi_data
import matplotlib.pyplot as plt


def get_best_archis_for_seed(
    seed,
    test_params,
    data_factory,
    gpus,
    methods_variant_params,
    checkpoint_dir,
):
    best_archis = {}
    for method_name, method_params in sorted(methods_variant_params.items()):
        method = archis.Method(method_params["method"])
        method_params = method_params.copy()
        del method_params["method"]

        _, trained_archi = xp.train_model(
            method,
            seed=seed,
            data_factory=data_factory,
            gpus=gpus,
            method_params=method_params,
            method_name=method_name,
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

    default_toy_data_file = "../configs/datasets/toy_3_blobs.json"

    parser = argparse.ArgumentParser(
        description="""
                    Train & evaluate domain adaptation architecture.
                    This script allows you to test the main methods
                    on any dataset of your choice with a limited
                    number of options.

                    Run::
                        python run_simple.py
                    
                    to run on a Blob dataset with 3 classes.

                    Run::
                        python run_simple.py -d ../configs/datasets/digits.json

                    to run on MNIST->USPS.

                    Have a look at the content of `../configs` for more details.
                    """,
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
    parser.add_argument("-g", "--gpu", nargs="*", help="id of gpu to use", type=int)

    parser.add_argument(
        "-n",
        "--netconf",
        help="path to config file defining network parameters and learning hyperparameters.",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    logging.info(f"Running with {args}")
    nseeds = args.nseeds
    data_params = xp.load_json_dict(args.dataconf)

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

    data_factory = DatasetFactory(data_params, data_path=args.data_path)
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

    results = xpr.XpResults.from_file(
        ["source acc", "target acc", "domain acc"], test_csv_file
    )
    do_plots = False

    methods_variant_params = xp.load_json_dict(args.method)

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
            method_name=method_name,
            method_params=method_params,
            checkpoint_dir=checkpoint_dir,
        )
        archis_res[method_name] = domain_archi
        if domain_archi is not None:
            do_plots = True
        results.to_csv(test_csv_file)

    print(results.get_data())

    logging.info("Recomputing context for best seed")
    mean_seed = results.get_mean_seed("target acc")
    archis_res = get_best_archis_for_seed(
        seed=mean_seed,
        test_params=test_params,
        data_factory=data_factory,
        gpus=args.gpu,
        methods_variant_params=methods_variant_params,
        checkpoint_dir=checkpoint_dir,
    )
    output_file_prefix = "_".join((output_file_prefix, str(mean_seed)))

    results.append_to_txt(
        filepath=os.path.join(output_dir, "all_res.txt"),
        test_params=test_params,
        nseeds=nseeds,
    )

    if do_plots:
        logging.getLogger("matplotlib").setLevel(logging.ERROR)

        for name, archi in archis_res.items():
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
