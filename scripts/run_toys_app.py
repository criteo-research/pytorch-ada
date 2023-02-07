import os
from pathlib import Path
import streamlit as st
from collections import defaultdict
import glob
import logging
import argparse

from ada.datasets.dataset_factory import DatasetFactory
import ada.utils.experimentation as xp
import ada.models.architectures as archis
import ada.utils.streamlit_configs as nc
import ada.utils.experimentation_results as xpr

from ada.utils.plotting import plot_archi_data
import matplotlib.pyplot as plt
import seaborn as sns


def get_archi_or_file_for_seed(
    seed,
    methods,
    file_prefix,
    test_params,
    data_factory,
    checkpoint_dir,
    gpus,
    fig_names,
):
    best_archis = {}
    st.write("Generating models for plots.")
    st.text(f"Looking for {file_prefix}_{seed}_<method>_*.png")
    howtext = st.text("")
    pgbar = st.progress(0)
    for i, method in enumerate(methods):
        if (
            data_factory.is_semi_supervised()
            and not archis.Method(method).is_fewshot_method()
        ):
            logging.warning(
                f"Skipping {method}: not suited for the semi-supervised setting."
            )
            continue
        png_files = glob.glob(f"{file_prefix}_{seed}_{method}_*.png")
        recompute = False
        for figname in fig_names:
            png_figs = glob.glob(f"{file_prefix}_{seed}_{method}_*{figname}*.png")
            if len(png_figs) == 0:
                recompute = True

        if recompute:
            howtext.text(f"Retraining {method}")
            trainee, trained_archi = xp.train_model(
                method,
                seed=seed,
                data_factory=data_factory,
                checkpoint_dir=checkpoint_dir,
                try_to_resume=True,
                gpus=gpus,
                **test_params,
            )
            best_archis[method] = trained_archi
        else:
            howtext.text("Restoring images")
            best_archis[method] = png_files
        pgbar.progress((i + 1) / len(methods))
    return best_archis


def run_loop(force_run=False):
    if auto_run or force_run:
        run_state.markdown(
            f":timer_clock: Running with settings: few={fewshot}, nseeds={nseeds}, methods={methods}."
        )
    else:
        run_state.markdown(
            f":warning: Click button on the left to run training & evaluation."
        )
        return

    data_factory = DatasetFactory(
        toy_params, data_path=args.data_path, n_fewshot=fewshot
    )
    os.makedirs(output_dir, exist_ok=True)
    test_hash = xp.param_to_hash(test_params)
    output_file_prefix = os.path.join(output_dir, test_hash)

    checkpoint_dir = os.path.join(output_dir, "checkpoints", test_hash)

    xp.record_hashes(
        os.path.join(args.outdir, "data_hashes.json"), data_hash, toy_params
    )
    xp.record_hashes(
        os.path.join(args.outdir, "test_hashes.json"), test_hash, test_params
    )

    test_csv_file = f"{output_file_prefix}.csv"
    results = xpr.XpResults.from_file(
        ["source acc", "target acc", "domain acc"], test_csv_file
    )

    archis_res = {}
    for m in methods:
        st.write(f"Learning {nseeds} x {m}")
        pgbar = st.progress(0)
        domain_archi = xp.loop_train_test_model(
            m,
            results,
            nseeds,
            test_csv_file,
            test_params=test_params,
            data_factory=data_factory,
            force_run=False,
            gpus=gpus,
            checkpoint_dir=checkpoint_dir,
            progress_callback=lambda percent: pgbar.progress(percent),
        )
        if domain_archi is not None:
            archis_res[m] = domain_archi
        results.to_csv(test_csv_file)

    st.header("Results summary")
    st.write(f"Read from {test_csv_file}")
    st.dataframe(results.get_data().groupby(["method", "split"]).mean())
    print(results.get_data())
    fig = plt.figure()
    ax = sns.catplot(
        x="method", y="target acc", data=results.get_data(), kind="swarm", hue="split"
    )
    plt.ylabel("Accuracy")
    st.pyplot()

    results.append_to_txt(
        filepath=os.path.join(output_dir, "all_res.txt"),
        test_params=test_params,
        nseeds=nseeds,
    )
    st.header("Plots")

    logging.info("Recomputing context for best seed")

    mean_seed = results.get_mean_seed("target acc")
    archis_res = get_archi_or_file_for_seed(
        seed=mean_seed,
        methods=methods,
        file_prefix=output_file_prefix,
        test_params=test_params,
        data_factory=data_factory,
        checkpoint_dir=checkpoint_dir,
        gpus=gpus,
        fig_names=fig_names,
    )
    output_file_prefix = f"{output_file_prefix}_{mean_seed}"

    from PIL import Image

    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    all_figs = defaultdict(dict)
    for name, res in archis_res.items():
        if isinstance(res, list):
            for fig_name in fig_names:
                for fig_file in res:
                    if "_".join(fig_name.split()) in fig_file:
                        all_figs[fig_name][name] = Image.open(fig_file)
        elif res is not None:
            figs = plot_archi_data(
                res,
                name,
                save_prefix=output_file_prefix,
                plot_f_lines="neurons" in fig_names,
                do_domain_boundary="domain boundary" in fig_names,
                plot_features=set(fig_names) & set(["PCA", "UMAP", "TSNE"]),
                do_entropy="entropy" in fig_names,
            )
            for fig_name, fig in figs.items():
                if fig_name in fig_names:
                    all_figs[fig_name][name] = fig
            plt.close("all")
        else:
            st.write(f"Cannot show result for {name}: {res}")

    for fig_name, figs in all_figs.items():
        st.subheader(fig_name)
        for meth_name, fig in figs.items():
            st.write(meth_name)
            if isinstance(fig, Image.Image):
                st.image(fig, use_column_width=True)
            else:
                st.pyplot(fig)

    logging.info(f"See results with prefix {output_file_prefix}")
    run_state.text(f"Done, see results in {output_file_prefix}.")


if __name__ == "__main__":
    st.title("Domain adaptation with toy data")
    st.write(
        "Easily create a toy dataset that helps you"
        " understand/trick domain-adaptive representations."
    )
    run_state = st.text("Running all...")

    auto_run = st.sidebar.button("Run")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    default_network_file = "../configs/toy_network.json"

    all_methods = [method.value for method in archis.Method]
    methods = st.sidebar.multiselect("Methods", all_methods, default=["Source"])

    parser = argparse.ArgumentParser(
        description="Train & evaluate domain adaptation architecture."
    )

    nseeds = st.sidebar.number_input(
        "Number of seeds", min_value=1, max_value=100, value=1
    )
    is_few_ratio = st.sidebar.checkbox("Use ratio for few-shot")
    if is_few_ratio:
        fewshot = st.sidebar.slider(
            "Ratio of labeled samples per class (few-shot)",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
        )
    else:
        fewshot = st.sidebar.number_input(
            "Number of labeled samples per class (few-shot)",
            min_value=0,
            max_value=10,
            value=0,
        )

    parser.add_argument(
        "-o",
        "--outdir",
        help="output directory name relative to root.",
        default=os.path.join(str(Path.home()), "sync", "ada", "toys_app"),
    )

    parser.add_argument(
        "-r",
        "--run",
        help="Override run button and run if true",
        action="store_true",
    )

    parser.add_argument(
        "--data_path",
        help="Full path to input data directory",
        type=str,
        default=os.path.join(str(Path.home()), "sync", "data"),
    )

    parser.add_argument("-g", "--gpu", help="id of gpu to use", type=int, default=-1)

    args = parser.parse_args()

    toy_params, data_hash = nc.configure_dataset(
        "../configs/datasets", on_sidebar=False
    )

    output_dir = os.path.join(args.outdir, data_hash)
    os.makedirs(os.path.join(args.outdir), exist_ok=True)

    network_params = nc.configure_network(default_network_file)

    gpus = [args.gpu] if args.gpu >= 0 else None

    test_params = network_params.copy()

    default_fig_names = [
        "classifier boundary",
        "PCA",
    ]
    all_figs = [
        "domain boundary",
        "entropy",
        "neurons",
        "TSNE",
        "UMAP",
    ] + default_fig_names
    fig_names = st.multiselect(
        "Choose your plots:", all_figs, default=default_fig_names
    )

    run_loop(args.run)  # with ugly use of global variables from main :-/
