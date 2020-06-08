import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import logging


PALETTES = [
    "deep",
    "pastel",
    "bright",
    "dark",
    "colorblind",
]
COLORS = [sns.color_palette(pal_name, 4) for pal_name in PALETTES]
DIV_PAL = sns.diverging_palette(180, 0, l=60, n=10)


def colored_scattered_plot2x2(
    X_s,
    X_t,
    y_sparse_train_s,
    y_sparse_train_t,
    figsize=(12, 6),
    set_aspect_equal=False,
):
    # scatter plot, dots colored by class value
    df_s = pd.DataFrame(
        dict(x=X_s[:, 0], y=X_s[:, 1], label=y_sparse_train_s.squeeze())
    )
    df_t = pd.DataFrame(
        dict(x=X_t[:, 0], y=X_t[:, 1], label=y_sparse_train_t.squeeze())
    )
    marker_s = "o"
    marker_t = "x"
    fig, ax = plt.subplots(figsize=figsize)
    grouped_s = df_s.groupby("label")
    grouped_t = df_t.groupby("label")
    if len(grouped_s) < 4:
        colors = COLORS
    else:
        colors = [sns.color_palette(pal_name, len(grouped_s)) for pal_name in PALETTES]
    for key, group in grouped_s:
        group.plot(
            ax=ax,
            kind="scatter",
            x="x",
            y="y",
            label=str(key) + "_source",
            color=colors[0][key],
            marker=marker_s,
        )
    for key, group in grouped_t:
        group.plot(
            ax=ax,
            kind="scatter",
            x="x",
            y="y",
            label=str(key) + "_target",
            color=colors[1][key],
            marker=marker_t,
        )
    if set_aspect_equal:
        ax.set_aspect("equal")
    fig.set_tight_layout(tight=None)
    return fig, ax


def plot_archi_data(
    domain_archi,
    tag,
    save_prefix=None,
    plot_features=None,
    plot_f_lines=False,
    do_domain_boundary=False,
    do_entropy=False,
    num_samples=600,
):
    """This method generates a series of figures depending on the model and
    on the dataset used.
    
    For toy data, more figures are available:

        - the classifier boundary
        - the domain boundary
        - the entropy values
        - the lines corresponding to the hidden neurons of the first feature layer
        - a PCA or TSNE or UMAP projection of the features

    For other datasets with more than 2 dimensions,
        - only the feature projections

    are available.

    Args:
        domain_archi (BaseAdaptTrainer):
            the trained architecture.
        tag (string): 
            the name of the method used both in the generated image titles and file names.
        save_prefix (string, optional): defaults to None.
            images will be saved to "{save_prefix}_{auto-gen-name}.png"
            If save_prefix is None, the images are not saved to disk.
        plot_features (bool): defaults to None
            None or string or list of strings from ("pca", "tsne", "umap")
        plot_f_lines (bool, optional): defaults to False.
            If True, plot the lines corresponding to the first neurons for 2D data. 
        do_domain_boundary (bool, optional): defaults to False
            If True, plot the domain boundary for 2D toy data.
        do_entropy (bool, optional): defaults to False
            If True, plots the level of entropy values between 0 and 1
        num_samples (int, optional): defaults to 600
            Number of random samples use for plotting

    """
    import torch

    dataset = domain_archi._dataset
    model = domain_archi.to("cpu")
    feat_extract = domain_archi.feat
    figs = {}

    num_src = num_samples // 2

    dl = iter(dataset.get_domain_loaders(batch_size=num_src))
    if dataset.is_semi_supervised():
        (X_source, y_source), (X_tl, y_tl), (X_target, y_target) = next(dl)
    else:
        (X_source, y_source), (X_target, y_target) = next(dl)

    dim = X_source.view(X_source.shape[0], -1).shape[1]
    h = 0.1

    if dim > 2:
        do_domain_boundary = False
        do_classifier_boundary = False
        do_entropy = False
        plot_f_lines = False
    else:
        do_classifier_boundary = True
        do_domain_boundary = (
            do_domain_boundary and not domain_archi.method.is_mmd_method()
        )

    if do_classifier_boundary:
        x_min, x_max = (
            np.minimum(X_source[:, 0].min(), X_target[:, 0].min()),
            np.maximum(X_source[:, 0].max(), X_target[:, 0].max()),
        )
        y_min, y_max = (
            np.minimum(X_source[:, 1].min(), X_target[:, 1].min()),
            np.maximum(X_source[:, 1].max(), X_target[:, 1].max()),
        )
        # increase margin
        x_min, x_max = 1.1 * x_min - 0.1, 1.1 * x_max + 0.1
        y_min, y_max = 1.1 * y_min - 0.1, 1.1 * y_max + 0.1

        fig, ax = colored_scattered_plot2x2(X_source, X_target, y_source, y_target)

        if dataset.is_semi_supervised():
            plt.scatter(X_tl[:, 0], X_tl[:, 1], c=np.array(COLORS[2])[y_tl])

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = np.c_[xx.ravel(), yy.ravel()]
        Z_torch = torch.from_numpy(np.atleast_2d(Z)).float()

        if not domain_archi.method.is_mmd_method():
            Z_feat, Z_class, D_class = model.forward(Z_torch)
        else:
            Z_feat, Z_class = model.forward(Z_torch)

        classe = Z_class.data.max(1)[1].numpy()

        classe = classe.reshape(xx.shape)
        n_classes = len(np.unique(y_source))
        plt.contour(xx, yy, classe, alpha=0.8, colors=COLORS[0], levels=n_classes - 2)
        plt.contourf(xx, yy, classe, alpha=0.2, colors=COLORS[0], levels=n_classes - 1)
        plt.title(f"{tag} Classifier boundary")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        if save_prefix is not None:
            plt.savefig(
                f"{save_prefix}_{tag}_classifier_boundary.png", bbox_inches="tight"
            )
        figs["classifier boundary"] = fig

    if do_entropy:
        from torch.nn import functional as F

        class_output = F.softmax(Z_class, dim=1)
        positives = torch.gt(class_output, 0.0).double().sum()
        loss_ent = (
            -1.0
            * torch.sum(class_output * (torch.log(class_output + 1e-9)), 1)
            .detach()
            .numpy()
        )
        entropy = loss_ent.reshape(xx.shape)
        fig, _ = colored_scattered_plot2x2(X_source, X_target, y_source, y_target)
        plt.contourf(xx, yy, entropy, alpha=0.2, colors=DIV_PAL, levels=10)
        plt.title(f"{tag} Entropy")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        if save_prefix is not None:
            plt.savefig(f"{save_prefix}_{tag}_entropy.png", bbox_inches="tight")

        figs[f"entropy"] = fig

    if do_domain_boundary:
        fig, _ = colored_scattered_plot2x2(X_source, X_target, y_source, y_target)

        dom_class = D_class.data.max(1)[1].numpy()
        dom_class = dom_class.reshape(xx.shape)
        plt.contour(xx, yy, dom_class, levels=[0], colors="black")
        plt.contourf(xx, yy, dom_class, alpha=0.2, colors=COLORS[0], levels=1)
        plt.title(f"{tag} Domain classifier boundary")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        if save_prefix is not None:
            plt.savefig(f"{save_prefix}_{tag}_domain_boundary.png", bbox_inches="tight")

        figs[f"domain boundary"] = fig

    if plot_f_lines and dim == 2:
        fig, ax = colored_scattered_plot2x2(X_source, X_target, y_source, y_target)
        W = feat_extract.feature[0].weight.data.numpy()
        B = feat_extract.feature[0].bias.data.numpy()
        for i in range(W.shape[0]):
            x0 = np.linspace(x_min, x_max)
            y0 = -W[i, 0] / W[i, 1] * x0 - B[i] / W[i, 1]
            ax.plot(x0, y0, alpha=0.3, c="black")
        ax.set_ylim([y_min, y_max])
        plt.title(f"{tag} Hidden neurons (first feature layer)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        if save_prefix is not None:
            plt.savefig(f"{save_prefix}_{tag}_neurons.png", bbox_inches="tight")

        figs[f"neurons"] = fig

    # Get embeddings
    if plot_features:
        if isinstance(plot_features, str):
            plot_features = [plot_features]

        X_source_map = feat_extract(X_source).data.numpy()
        X_target_map = feat_extract(X_target).data.numpy()

        emb_all = np.vstack([X_source_map, X_target_map])

        for proj in plot_features:
            comment = f"{proj.upper()} projection of feature layer"

            if proj.upper() == "PCA":
                from sklearn.decomposition import PCA

                pca = PCA(n_components=2)
                fea_plot = pca.fit_transform(emb_all)
            elif proj.upper() == "TSNE":
                from sklearn.manifold import TSNE

                tsne = TSNE(n_components=2, init="random", random_state=9365)
                fea_plot = tsne.fit_transform(emb_all)
            elif proj.upper() == "UMAP":
                import umap

                um = umap.UMAP(n_neighbors=10, min_dist=0.1, metric="euclidean")
                um.fit(emb_all)
                fea_plot = um.transform(emb_all)

            num = X_source.shape[0]

            fig, _ = colored_scattered_plot2x2(
                fea_plot[:num, :], fea_plot[num:, :], y_source.numpy(), y_target.numpy()
            )
            plt.title(f"{tag} {comment}")
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            if len(np.unique(y_source)) < 4:
                # legend becomes useless when there are many classes
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
            if save_prefix is not None:
                plt.savefig(
                    f"{save_prefix}_{tag}_features_{proj.upper()}.png",
                    bbox_inches="tight",
                )

            figs[proj.upper()] = fig
    return figs
