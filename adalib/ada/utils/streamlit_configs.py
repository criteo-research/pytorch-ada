import os
import streamlit as st
import glob
import json
import numpy as np
import ada.utils.experimentation as xp
from copy import deepcopy


def configure_network(default_file, on_sidebar=True):
    network_params = xp.load_json_dict(default_file)
    stmod = st.sidebar if on_sidebar else st
    stmod.header("Network configuration")
    stmod.subheader("Learning parameters")
    stmod.markdown("$\\lambda$ controls the weight of the critic")
    adapt_lambda = stmod.checkbox(
        "Use adaptive lambda", value=network_params["train_params"]["adapt_lambda"]
    )
    lambda_init = stmod.number_input(
        "Final (max) lambda",
        value=float(network_params["train_params"]["lambda_init"]),
        step=1e-3,
        format="%.1f",
    )
    adapt_lr = stmod.checkbox(
        "Use adaptive learning rate", value=network_params["train_params"]["adapt_lr"]
    )
    init_lr = stmod.number_input(
        "Initial learning rate",
        value=float(network_params["train_params"]["init_lr"]),
        step=1e-4,
        format="%.4f",
    )
    nb_init_epochs = stmod.number_input(
        "Number of warmup epochs",
        min_value=0,
        value=network_params["train_params"]["nb_init_epochs"],
    )
    nb_adapt_epochs = stmod.number_input(
        "Number of adaptation epochs",
        min_value=1,
        value=network_params["train_params"]["nb_adapt_epochs"],
    )
    stmod.subheader("Architecture details")
    stmod.markdown(
        "Choose the width of the Feature extractor hidden layer. The size of the "
        "final layer width is used for both Task Classifier and Critic. "
        "The Critic can have more than one layer, all of same width."
    )
    hidden_size = stmod.text_input(
        "Feature hidden layers",
        value=network_params["archi_params"]["feature"]["hidden_size"],
    )
    hidden_size = json.loads(hidden_size)
    critic_layers = stmod.text_input(
        "Critic hidden layers",
        value=network_params["archi_params"]["critic"]["hidden_size"],
    )
    critic_layers = json.loads(critic_layers)

    network_params["train_params"]["adapt_lambda"] = adapt_lambda
    network_params["train_params"]["lambda_init"] = lambda_init
    network_params["train_params"]["adapt_lr"] = adapt_lr
    network_params["train_params"]["init_lr"] = init_lr
    network_params["train_params"]["nb_init_epochs"] = nb_init_epochs
    network_params["train_params"]["nb_adapt_epochs"] = nb_adapt_epochs
    network_params["archi_params"]["feature"]["hidden_size"] = hidden_size
    network_params["archi_params"]["critic"]["hidden_size"] = critic_layers
    return network_params


def configure_dataset(default_dir, on_sidebar=True):
    stmod = st.sidebar if on_sidebar else st
    stmod.header("Dataset")
    json_files = glob.glob(f"{default_dir}/*.json")
    all_params_files = [(f, xp.load_json_dict(f)) for f in json_files]
    toy_files = [
        f for (f, p) in all_params_files if p.get("dataset_group", "none") == "toy"
    ]
    dataset = stmod.selectbox("Dataset", toy_files, index=0)
    default_params = xp.load_json_dict(dataset)
    if on_sidebar:
        return default_params

    toy_params = deepcopy(default_params)

    # centers position
    default_centers = np.array([[-0.5, 0.0], [0.5, 0]])
    param_centers = default_params["cluster"].get("centers", default_centers.tolist())
    new_centers_st = stmod.text_input(
        "Position of class centers (source)", param_centers
    )
    new_centers = json.loads(new_centers_st)
    n_clusters = len(new_centers)
    stmod.markdown(f"{n_clusters} classes.")
    toy_params["cluster"]["centers"] = new_centers
    toy_params["cluster"]["n_clusters"] = n_clusters

    # centers radii
    radius0 = default_params["cluster"]["radius"]
    same_radius = stmod.checkbox(
        "Use same variance everywhere (class/dimension)",
        value=isinstance(radius0, float),
    )
    if same_radius:
        if not isinstance(radius0, float):
            radius0 = np.array(radius0).flatten()[0]
        radius = stmod.number_input(
            "Class variance",
            step=10 ** (np.floor(np.log10(radius0))),
            value=radius0,
            format="%.4f",
        )
        toy_params["cluster"]["radius"] = radius
    else:
        if isinstance(radius0, float):
            radii = (np.ones_like(new_centers) * radius0).tolist()
        else:
            radii = radius0
        new_radius_st = stmod.text_input(
            "Variance of each class along each dimension", radii
        )
        new_radius = json.loads(new_radius_st)
        shape_variance = np.array(new_radius).shape
        shape_clusters = np.array(new_centers).shape
        if shape_variance == shape_clusters:
            stmod.markdown(
                ":heavy_check_mark: Shape of variance values matches the shape of clusters."
            )
        else:
            stmod.markdown(
                ":warning: Warning: Shape of variances doesn't match the shape of clusters."
            )

        toy_params["cluster"]["radius"] = new_radius

    # class balance
    proba_classes = default_params["cluster"]["proba_classes"]
    if n_clusters == 2:
        new_proba_classes = stmod.number_input(
            "Probability of class 1",
            step=10 ** (np.floor(np.log10(proba_classes))),
            value=proba_classes,
            format="%.4f",
        )
    else:
        new_proba_classes_st = stmod.text_input(
            "Weight or probability of each class (will be normalized to sum to 1)",
            proba_classes,
        )
        new_proba_classes = json.loads(new_proba_classes_st)
        nb_probas = len(new_proba_classes)
        if nb_probas == n_clusters:
            stmod.markdown(
                ":heavy_check_mark: class probas values matches the number of clusters."
            )
        else:
            stmod.markdown(
                ":warning: Warning: class probas values don't match the number of clusters."
            )
    toy_params["cluster"]["proba_classes"] = new_proba_classes

    # target shift
    default_cond_shift = default_params["shift"]["data_shift"]
    if n_clusters == 2:
        cond_shift = stmod.checkbox(
            "Class-conditional shift", value="cond" in default_cond_shift
        )
    else:
        cond_shift = False

    if cond_shift:
        rotation0 = default_params["shift"]["re"]
        if isinstance(rotation0, float):
            default_r0 = rotation0
            default_r1 = rotation0
        else:
            default_r0 = default_params["shift"]["re"][0]
            default_r1 = default_params["shift"]["re"][1]
        re0 = stmod.slider(
            "Rotation class 0", min_value=-np.pi, max_value=np.pi, value=default_r0,
        )
        re1 = stmod.slider(
            "Rotation class 1", min_value=-np.pi, max_value=np.pi, value=default_r1,
        )
        transl0 = default_params["shift"]["te"]
        if isinstance(transl0, float):
            default_t0 = transl0
            default_t1 = transl0
        else:
            default_t0 = default_params["shift"]["te"][0]
            default_t1 = default_params["shift"]["te"][1]
        te0 = stmod.slider(
            "Translation class 0", min_value=-3.0, max_value=3.0, value=default_t0,
        )
        te1 = stmod.slider(
            "Translation class 1", min_value=-3.0, max_value=3.0, value=default_t1,
        )
        toy_params["shift"]["re"] = [re0, re1]
        toy_params["shift"]["te"] = [te0, te1]
    else:
        re = stmod.slider(
            "Rotation",
            min_value=-np.pi,
            max_value=np.pi,
            value=default_params["shift"]["re"],
        )
        te = stmod.slider(
            "Translation",
            min_value=-3.0,
            max_value=3.0,
            value=default_params["shift"]["te"],
        )
        toy_params["shift"]["re"] = re
        toy_params["shift"]["te"] = te

    test_view_data(toy_params)

    # choose a new (unique) name for the dataset and save
    data_hash = xp.param_to_hash(toy_params)
    default_hash = xp.param_to_hash(default_params)
    default_name = toy_params["dataset_name"]
    if default_hash != data_hash:
        if toy_params["dataset_name"] == default_params["dataset_name"]:
            default_name = data_hash

    data_name = st.text_input("Choose a (unique) name for your dataset", default_name)
    data_name = data_name.replace(" ", "_")
    toy_params["dataset_name"] = data_name
    data_file = os.path.join(default_dir, f"{data_name}.json")
    if os.path.exists(data_file):
        st.text(f"Data set with this name exists! {data_file}")
    else:
        if st.button("Save dataset"):
            with open(data_file, "w") as fd:
                fd.write(json.dumps(toy_params))
                default_params = deepcopy(toy_params)
            st.text(f"Configuration saved to {data_file}")

    return toy_params, data_hash


def test_view_data(data_params):
    from ada.utils.plotting import colored_scattered_plot2x2
    from ada.datasets.toys import CausalBlobs, get_datashift_params

    target_shift = get_datashift_params(**data_params["shift"])
    source_data = CausalBlobs(
        ".tmp_view",
        n_samples=data_params["n_samples"],
        transform=get_datashift_params("no_shift"),
        cluster_params=data_params["cluster"],
    )
    target_data = CausalBlobs(
        ".tmp_view",
        n_samples=data_params["n_samples"],
        transform=target_shift,
        cluster_params=data_params["cluster"],
    )
    X_s, y_s = source_data.data, source_data.targets
    X_t, y_t = target_data.data, target_data.targets

    fig, ax = colored_scattered_plot2x2(X_s, X_t, y_s, y_t, set_aspect_equal=True)
    fig.set_tight_layout(tight=None)
    st.write("View data")
    st.pyplot(fig)

    source_data.delete_from_disk()
    target_data.delete_from_disk()
