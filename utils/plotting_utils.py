from plothist import (
    make_hist,
    plot_hist,
    plot_error_hist,
    plot_comparison,
    create_comparison_figure,
    get_color_palette,
    plot_hist_uncertainties,
)
import boost_histogram as bh

import numpy as np
import pylab as plt
import os
from utils.generate_html import generate_html_from_dir, show_url

from utils.base_utils import combine_weight_columns

# import utils


def get_bin_centers_and_widths(bins):
    bins = np.array(bins)
    return (bins[:-1] + bins[1:]) / 2, np.diff(bins)


def flatten(arr):
    if hasattr(arr, "flatten"):
        return arr.flatten()
    elif hasattr(arr, "to_numpy"):
        return arr.to_numpy().flatten()


# def get_weights_from(y_train, y_test):
#     """
#     Calculate the sample weights

#     Parameters:
#     y_train (array-like): The target values for the training data.
#     y_test (array-like): The target values for the testing data.

#     Returns:
#     tuple: A tuple containing the weights for the testing data and the weights for the training data.
#     """

#     y_test = flatten(y_test)
#     y_train = flatten(y_train)

#     n_test = len(y_test)
#     n_ones_test = np.count_nonzero(y_test)
#     n_zeros_test = n_test - n_ones_test

#     n_train = len(y_train)
#     n_ones_train = np.count_nonzero(y_train)
#     n_zeros_train = n_train - n_ones_train

#     weights_test = np.ones(n_test) / n_test
#     weights_train = np.ones(n_train)

#     print(n_ones_test)

#     # (2 * n_test_one / n_test)

#     weights_train[y_train == 0] = (2 * n_zeros_test / n_test) / n_train
#     weights_train[y_train == 1] = (2 * n_ones_test / n_test) / n_train

#     return weights_test, weights_train


# def get_weights_from(y, y_reference, normalize=True):
#     """
#     Calculate the sample weights

#     Parameters:
#     y (array-like): The target values for the data.
#     y_reference (array-like): The reference values used to calculate the weights.

#     Returns:
#     tuple: A tuple containing the weights for the data and the weights for the reference data.
#     """
#     n = len(y)
#     n_ones = np.count_nonzero(y)
#     n_zeros = n - n_ones

#     n_ref = len(y_ref)
#     n_ones_ref = np.count_nonzero(y_ref)
#     n_zeros_ref = n_ref - n_ones_ref

#     weights_reference = np.ones(n)
#     weights = np.ones(n)

#     weights[y == 0] = (2 * n_zeros_ref / n_ref)
#     weights[y == 1] = (2 * n_ones_ref / n_ref)

#     if normalize:
#         weights /= n
#         weights_reference /= n_ones_ref

#     return weights, weights_reference


def plt_subplots(**kwargs):
    # plt.subplots(nrows=2, ncols=1, figsize=(6, 5), sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace':0.05})
    kwargs.setdefault("figsize", (6, 5))
    kwargs.setdefault("sharex", True)
    kwargs.setdefault("height_ratios", [3, 1])
    # kwargs.setdefault('hspace', 0.05)
    kwargs.setdefault("gridspec_kw", {})
    # kwargs['gridspec_kw'].setdefault('height_ratios', [3, 1])
    kwargs["gridspec_kw"].setdefault("hspace", 0.05)
    kwargs.setdefault("nrows", 2)
    kwargs.setdefault("ncols", 1)
    print(kwargs)
    fig, axs = plt.subplots(**kwargs)
    return fig, axs


def get_sample_weights_from_reference(y, y_ref, normalize=True):
    """
    Calculate the sample weights for sample y based on the population of sample y_ref

    Parameters:
    y (array-like): The target values for the data.
    y_reference (array-like): The reference values used to calculate the weights.

    Returns:
    tuple: A tuple containing the weights for the data and the weights for the reference data.
    """

    n = len(y)
    n_ones = np.count_nonzero(y)
    n_zeros = n - n_ones

    n_ref = len(y_ref)
    n_ones_ref = np.count_nonzero(y_ref)
    n_zeros_ref = n_ref - n_ones_ref

    weights_reference = np.ones(n_ref)
    weights = np.ones(n)

    weights[y == 0] = (n_zeros_ref / n_ref) * (n / n_zeros)
    weights[y == 1] = (n_ones_ref / n_ref) * (n / n_ones)

    if normalize:
        weights /= n
        weights_reference /= n_ref

    return weights, weights_reference


def calculate_sample_weights(y_1, y_2):
    """
    Calculate the sample weights for sample y_1 based on the population of sample y_2

    Parameters:
    y_1 (array-like): The sample for which weights need to be calculated.
    y_2 (array-like): The population sample used to calculate the weights.

    Returns:
    tuple: A tuple containing the weights for y_1 and y_2 respectively.
    """

    y_1 = flatten(y_1)
    y_2 = flatten(y_2)

    n_1 = len(y_1)
    n_ones_1 = np.count_nonzero(y_1)
    n_zeros_1 = n_1 - n_ones_1

    n_2 = len(y_2)
    n_ones_2 = np.count_nonzero(y_2)
    n_zeros_2 = n_2 - n_ones_2

    weights_1 = np.ones(n_1) / n_1
    weights_2 = np.ones(n_2)

    print(n_ones_1)

    weights_2[y_2 == 0] = (2 * n_zeros_1 / n_1) / n_2
    weights_2[y_2 == 1] = (2 * n_ones_1 / n_1) / n_2

    return weights_1, weights_2


def plot_roc(fig_ax=None, truth=None, pred=None, weights=None, **plot_kwargs):
    import sklearn

    fig, ax = fig_ax if fig_ax else plt.subplots()

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(truth, pred, sample_weight=weights)
    area = np.trapz(tpr, fpr)
    area_label = "AUC=%s" % round(area, 3)

    if "label" in plot_kwargs:
        plot_kwargs["label"] += f" ({area_label})"
    else:
        plot_kwargs["label"] = f"{area_label}"

    line = ax.plot(fpr, tpr, **plot_kwargs)
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    return dict(
        fig=fig, ax=ax, fpr=fpr, tpr=tpr, thresholds=thresholds, area=area, line=line
    )


def plot_prc(fig_ax=None, truth=None, pred=None, weights=None, **plot_kwargs):
    # write the docstring for this function
    """
    plot the precision-recall curve

    Parameters
    ----------
    fig_ax : None or tuple
        if None, a new figure and axis will be created, otherwise a tuple of (fig, ax) should be given
    truth : array-like
        the true labels
    pred : array-like

    Returns
    -------
    dict
        a dictionary containing the figure, axis, precision, recall, thresholds, area, and line

    """

    import sklearn
    from sklearn.metrics import precision_recall_curve

    fig, ax = fig_ax if fig_ax else plt.subplots()

    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
        truth, pred, sample_weight=weights
    )
    ap = round(
        sklearn.metrics.average_precision_score(truth, pred, sample_weight=weights), 3
    )
    ap_label = "Average Precision = %s" % round(ap, 3)

    if "label" in plot_kwargs:
        plot_kwargs["label"] += f" ({ap_label})"
    else:
        plot_kwargs["label"] = f"{ap_label}"

    line = ax.plot(recall, precision, **plot_kwargs)
    ax.set_ylabel("Precision")
    ax.set_xlabel("Recall")
    return dict(
        fig=fig,
        ax=ax,
        precision=precision,
        recall=recall,
        thresholds=thresholds,
        ap=ap,
        line=line,
    )


def compare_test_train(
    df_test,
    df_train=None,
    variable="pred",
    truth_var="truth",
    weight_var=None,
    bins=50,
    range=(-0.1, 1.1),
    comparison="pull",
    y_label="normalized events",
    y_label_ratio=None,
    y_scale=None,
    x_label=None,
    x_scale=None,
    plot_path=None,
    reweight_train_to_test=True,
):
    """
    Compare a variable, for instance the prediction, in the test and train datasets.

    """
    include_train = df_train is not None

    hists = []
    labels = []
    colors = []

    n_test = len(df_test)
    weights = np.ones(n_test) / n_test

    truth = df_test[truth_var]
    n_ones_test = np.count_nonzero(truth)
    n_zeros_test = n_test - n_ones_test

    sample_weights_test = combine_weight_columns(df_test, weight_var)
    # sample_weights_test = df_test[weight_var] if weight_var else np.ones(len(df_test))
    weights *= sample_weights_test

    # Test plots
    h1 = make_hist(
        df_test[variable][truth == 1],
        bins=bins,
        range=range,
        weights=weights[truth == 1],
    )
    h2 = make_hist(
        df_test[variable][truth == 0],
        bins=bins,
        range=range,
        weights=weights[truth == 0],
    )
    hists += [h1, h2]
    labels += ["test (pos)", "test (neg)"]
    colors += ["C0", "C1"]

    if include_train:
        n_train = len(df_train)
        truth = df_train[truth_var]

        # n_ones_train  = np.count_nonzero(train)
        # n_zeros_train = n_train - n_ones_train
        if reweight_train_to_test:
            weights_train, weights_test = get_sample_weights_from_reference(
                df_train[truth_var], df_test[truth_var]
            )
        else:
            weights_train = np.ones(n_train) / n_train

        weights = weights_train
        # sample_weights_train = (
        #     df_train[weight_var] if weight_var else np.ones(len(df_train))
        # )
        sample_weights_train = combine_weight_columns(df_train, weight_var)
        weights *= sample_weights_train

        h3 = make_hist(
            df_train[variable][df_train[truth_var] == 1],
            bins=bins,
            range=range,
            weights=weights[truth == 1],
        )
        h4 = make_hist(
            df_train[variable][df_train[truth_var] == 0],
            bins=bins,
            range=range,
            weights=weights[truth == 0],
        )
        hists += [h3, h4]
        labels += ["train (pos)", "train (neg)"]
        colors += ["C0", "C1"]
    hists_dict = dict(
        zip([l.replace(" (", "_").replace(")", "") for l in labels], hists)
    )
    # fig, axes = create_comparison_figure(
    #     figsize=(6, 6), nrows=2, gridspec_kw={"height_ratios": [5, 1]},
    # )
    fig, (ax_main, ax_comparison) = plt.subplots(
        nrows=2, gridspec_kw=dict(height_ratios=[5, 2]), sharex=True
    )
    fig.subplots_adjust(hspace=0.11)

    hist_kw = dict(histtype="step", linewidth=1.2)

    for ihist, h in enumerate(hists[:2]):
        plot_error_hist(h, label=labels[ihist], color=colors[ihist], ax=ax_main)

    if include_train:
        hist_kw.update(histtype="stepfilled", alpha=0.2)
        for ihist, h in enumerate(hists[2:], 2):
            print(ihist, colors, labels)
            plot_hist(
                h, label=labels[ihist], color=colors[ihist], ax=ax_main, **hist_kw
            )

        plot_comparison(
            h1, h3, ax=ax_comparison, comparison=comparison, color=colors[0], alpha=0.3
        )
        plot_comparison(
            h2, h4, ax=ax_comparison, comparison=comparison, color=colors[1], alpha=0.3
        )

    else:
        plot_comparison(
            h1, h2, ax=ax_comparison, comparison=comparison, color=colors[1], alpha=0.3
        )

        # fig.subplots_adjust(hspace=0.11)

    # print(a)

    y_label_size = 15
    ax_main.set_ylabel(y_label, size=y_label_size)
    if y_label_ratio:
        ax_comparison.set_ylabel(y_label_ratio, size=y_label_size)

    ax_comparison.set_xlabel(x_label if x_label else variable)

    ymin, ymax = ax_main.get_ylim()
    y_fact = 0.3 * ymax
    if y_scale:
        ax_main.set_yscale(y_scale)
        if y_scale == "log":
            if ymin == 0:
                ymin = 1e-10
            y_fact = 0.15 * np.log10(ymax / ymin)
    if x_scale:
        ax_main.set_xscale(x_scale)

    ax_main.legend(loc="upper left", ncols=2)

    ax_main.set_ylim(ymax=ymax + y_fact)
    fig.align_ylabels()
    if plot_path:
        savefig(fig, plot_path, formats=["png", "pdf"])
    return dict(fig=fig, ax_main=ax_main, ax_comp=ax_comparison, **hists_dict)


def get_cumsum_frac(x, bins=50, range=(0, 1), density=True):
    weights = 1 if not density else 1.0 / len(x)
    h = make_hist(x, bins=bins, range=range, weights=weights)

    bc, edges = h.to_numpy()
    cs = np.cumsum(bc)

    hcs = bh.Histogram(h.axes[0])
    hcs[:] = cs

    return hcs


# compare_preds(df_test, df_train, variable='pred',  range=(0,1), comparison="pull", y_label="normalized events", y_label_ratio="pull", x_label=None, y_scale='log' )
def savefig(
    fig, plot_path, formats=["png", "pdf"], copy_php=False, verbose=True, **kwargs
):
    kwargs.setdefault("dpi", 100)
    kwargs.setdefault("bbox_inches", "tight")
    print(kwargs)

    if "{" in plot_path:
        raise ValueError("plot_path needs more formating! %s" % plot_path)

    plot_dir_base = os.path.dirname(plot_path)
    plot_fmt = os.path.splitext(plot_path)[-1].replace(".", "")
    # plot_file = os.path.basename(plot_path)
    # plot_name_base, plot_fmt = os.path.splitext(plot_file)
    # plot_path_base = f'{plot_dir_base}/{plot_name_base}'

    if plot_dir_base and plot_dir_base != ".":
        if not os.path.isdir(plot_dir_base) and verbose:
            print("making dir:", plot_dir_base)
        os.makedirs(plot_dir_base, exist_ok=True)

    if copy_php:
        copyIndexPHP(plot_dir_base)

    if plot_fmt:
        formats = [plot_fmt]

    for fmt in formats:
        plot_path_ = plot_path if plot_fmt else plot_path + f".{fmt}"
        fig.savefig(plot_path_, **kwargs)
        if verbose:
            print("plot saved in: %s" % plot_path_)
        show_url(plot_path)
