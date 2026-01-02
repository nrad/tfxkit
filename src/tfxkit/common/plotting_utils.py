from tfxkit.common import base_utils
from tfxkit.common.base_utils import logger
import pandas as pd
import logging

# logger = base_utils.logging.getLogger(__name__)
logger = logging.getLogger(__name__)

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


def get_bin_centers_and_widths(bins):
    bins = np.array(bins)
    return (bins[:-1] + bins[1:]) / 2, np.diff(bins)


def plt_subplots(**kwargs):
    kwargs.setdefault("figsize", (6, 5))
    kwargs.setdefault("sharex", True)
    kwargs.setdefault("nrows", 2)
    kwargs.setdefault("ncols", 1)
    if kwargs["nrows"] == 2:
        kwargs.setdefault("height_ratios", [3, 1])
    # kwargs.setdefault('hspace', 0.05)
    kwargs.setdefault("gridspec_kw", {})
    # kwargs['gridspec_kw'].setdefault('height_ratios', [3, 1])
    kwargs["gridspec_kw"].setdefault("hspace", 0.05)
    # print(kwargs)    
    logger.debug(f"Creating subplots with kwargs: {kwargs}")
    fig, axs = plt.subplots(**kwargs)
    return fig, axs


def fix_train_history_df(df_hist):
    for col in df_hist.columns:
        if df_hist[col].dtype == object:
            expanded_col = pd.DataFrame(df_hist[col].tolist())
            expanded_col.columns = [f"{col}_{i}" for i in expanded_col.columns]
            df_hist.drop(col, axis=1, inplace=True)
            df_hist = pd.concat([df_hist, expanded_col], axis=1)
    return df_hist


def plot_history(
    history,
    ylim=None,
    xlabel="Epoch",
    ylabel="",
    plot_kwargs={},
    keys=None,
    plot_path=None,
):
    history = getattr(history, "history", history)
    df_hist = pd.DataFrame(history)
    df_hist = fix_train_history_df(df_hist)

    df_columns = df_hist.columns.tolist()
    keys = df_columns if keys is None else keys

    metrics = [k for k in keys if k in keys and not k.startswith("val_")]
    val_metrics = [f"val_{k}" for k in keys if f"val_{k}" in df_columns]

    fig, ax = plt.subplots()
    df_hist[metrics].plot(xlabel=xlabel, ylabel=ylabel, ylim=ylim, ax=ax, **plot_kwargs)
    if val_metrics:
        plt.gca().set_prop_cycle(None)
        df_hist[val_metrics].plot(style="--", ax=ax, **plot_kwargs)
        ax.legend(ncol=2, loc="upper right")
    if plot_path:
        save_fig(fig, plot_path, formats=["png", "pdf"])
    return fig, ax, df_hist


def make_classwise_hist(
    df, variable="pred", label_column="truth", weights=None, bins=50, range=(-0.1, 1.1)
):
    """
    Get the histogram of predictions for a given variable in the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing the predictions.
    variable (str): The column name for the predictions.
    label_column (str): The column name for the true labels.
    weights (str or array): The column name for the weights or a array-like of weights.
    bins (int): Number of bins for the histogram.
    range (tuple): Range of values for the histogram.

    Returns:
    tuple: A tuple containing the histogram and bin edges.
    """

    truth = df[label_column]
    cls0 = truth == 0
    cls1 = truth == 1

    weights = base_utils.combine_weight_columns(df, weights)

    hist_kwargs = dict(bins=bins, range=range)
    h0 = make_hist(df[variable][cls0], weights=weights[cls0], **hist_kwargs)
    h1 = make_hist(df[variable][cls1], weights=weights[cls1], **hist_kwargs)

    return h0, h1


def plot_classwise_hist(
    df_test,
    df_train=None,
    variable="pred",
    label_column="truth",
    weight_column=None,
    weight_column_train=None,
    bins=50,
    range=(-0.1, 1.1),
    comparison="pull",
    plot_path=None,
    query=None,
    **kwargs,
    # y_label="normalized events",
    # y_label_ratio=None,
    # y_scale=None,
    # x_label=None,
    # x_scale=None,
    # reweight_train_to_test=True,
):
    """
    Compare a variable, for instance the prediction, in the test and train datasets and for positive and negative classes
    Parameters
    ----------
    df_test : DataFrame
        The test dataset
    df_train : DataFrame, optional
        The train dataset, by default None
    variable : str, optional
        The variable to plot, by default "pred"
    label_column : str, optional
        The column name for the true labels, by default "truth"
    weight_column : str or array-like, optional
        The column name for the weights in the test dataset or an array-like of weights, by
        default None
    weight_column_train : str or array-like, optional
        The column name for the weights in the train dataset or an array-like of weights, by
        default same as weight_column. Note that this may be different than the weight used
        during the training. This is the weight to make the train dataset comparable to the
        test dataset.
    """
    include_train = df_train is not None

    hists = []
    labels = []
    colors = []

    n_test = len(df_test)

    pred_kwargs = dict(
        variable=variable,
        label_column=label_column,
        bins=bins,
        range=range,
    )
    if query:
        df_test = df_test.query(query)

    logger.debug(f"Plotting classwise hist for variable: {variable}")
    logger.debug(f"weight_column: {weight_column}")
    logger.debug(f"using kwargs: {pred_kwargs}")
    htest0, htest1 = make_classwise_hist(
        df_test,
        weights=weight_column,
        **pred_kwargs,
    )
    logger.debug(f"Test Histograms:\n\
                 negative class:\n{htest0}\n\
                 positive class:\n{htest1}")

    hists += [htest1, htest0]
    labels += ["test (pos)", "test (neg)"]
    colors += ["C0", "C1"]

    if include_train:
        if query:
            df_train = df_train.query(query)
        htrain0, htrain1 = make_classwise_hist(
            df_train,
            weights=(
                weight_column if weight_column_train is None else weight_column_train
            ),
            **pred_kwargs,
        )

        hists += [htrain1, htrain0]
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
            #print(ihist, colors, labels)
            plot_hist(
                h, label=labels[ihist], color=colors[ihist], ax=ax_main, **hist_kw
            )

        comp0 = plot_comparison(
            htest0,
            htrain0,
            ax=ax_comparison,
            comparison=comparison,
            color=colors[0],
            alpha=0.3,
        )
        comp1 = plot_comparison(
            htest1,
            htrain1,
            ax=ax_comparison,
            comparison=comparison,
            color=colors[1],
            alpha=0.3,
        )
        hists_dict.update({"comp0": comp0, "comp1": comp1})

    else:
        comparison = plot_comparison(
            htest0,
            htest1,
            ax=ax_comparison,
            comparison=comparison,
            color=colors[1],
            alpha=0.3,
        )
        hists_dict["comparison_test"] = comparison

        # fig.subplots_adjust(hspace=0.11)

    # print(a)
    plot_kwargs = dict(
        x_label=variable,
        y_label="normalized events",
        y_label_ratio="pull",
        x_scale="linear",
        y_scale="log",
        y_label_size=15,
        y_min_ratio=None,
        y_max_ratio=None,
    )
    plot_kwargs.update(kwargs)
    logger.debug(f"Plotting with kwargs: {plot_kwargs}")

    # y_label_size = 15
    # ax_main.set_ylabel(y_label, size=y_label_size)
    ax_main.set_ylabel(plot_kwargs["y_label"], size=plot_kwargs["y_label_size"])
    if plot_kwargs.get("y_label_ratio"):
        ax_comparison.set_ylabel(
            plot_kwargs["y_label_ratio"], size=plot_kwargs["y_label_size"]
        )

    ax_comparison.set_xlabel(plot_kwargs["x_label"])

    ymin, ymax = ax_main.get_ylim()
    y_fact = 0.3 * ymax
    y_scale = plot_kwargs.get("y_scale")
    x_scale = plot_kwargs.get("x_scale")
    if y_scale:
        ax_main.set_yscale(y_scale)
        if y_scale == "log":
            if ymin == 0:
                ymin = 1e-10
            y_fact = 0.2 * np.log10(ymax / ymin)
    if x_scale:
        ax_main.set_xscale(x_scale)

    ax_main.legend(loc="upper left", ncols=2)
    ax_main.set_ylim(ymax=ymax + y_fact)

    ax_comparison.set_ylim(
        ymin=plot_kwargs.get("y_min_ratio"), ymax=plot_kwargs.get("y_max_ratio")
    )
    fig.align_ylabels()
    if plot_path:
        save_fig(fig, plot_path, formats=["png", "pdf"])
    return dict(fig=fig, ax_main=ax_main, ax_comp=ax_comparison, **hists_dict)


##
##
##


def plot_roc(fig_ax=None, truth=None, pred=None, weights=None, **plot_kwargs):
    import sklearn.metrics

    fig, ax = fig_ax if fig_ax else plt.subplots()

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(truth, pred, sample_weight=weights)
    area = np.trapezoid(tpr, fpr)
    area_label = "AUC=%s" % round(area, 3)

    if "label" in plot_kwargs:
        plot_kwargs["label"] += f" ({area_label})"
    else:
        plot_kwargs["label"] = f"{area_label}"

    line = ax.plot(fpr, tpr, **plot_kwargs)
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    ax.legend()
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


def get_cumsum_frac(x, bins=50, range=(0, 1), density=True):
    weights = 1 if not density else 1.0 / len(x)
    h = make_hist(x, bins=bins, range=range, weights=weights)

    bc, edges = h.to_numpy()
    cs = np.cumsum(bc)

    hcs = bh.Histogram(h.axes[0])
    hcs[:] = cs

    return hcs


# compare_preds(df_test, df_train, variable='pred',  range=(0,1), comparison="pull", y_label="normalized events", y_label_ratio="pull", x_label=None, y_scale='log' )
def save_fig(
    fig, plot_path, formats=["png", "pdf"], copy_php=False, **kwargs
):
    if plot_path is None:
        return
    kwargs.setdefault("dpi", 100)
    kwargs.setdefault("bbox_inches", "tight")
    # print(kwargs)

    if "{" in plot_path:
        raise ValueError("plot_path needs more formating! %s" % plot_path)

    plot_dir_base = os.path.dirname(plot_path)
    plot_fmt = os.path.splitext(plot_path)[-1].replace(".", "")
    # plot_file = os.path.basename(plot_path)
    # plot_name_base, plot_fmt = os.path.splitext(plot_file)
    # plot_path_base = f'{plot_dir_base}/{plot_name_base}'

    if plot_dir_base and plot_dir_base != ".":
        logger.debug(f"Ensuring plot directory exists: {plot_dir_base}"  )
        os.makedirs(plot_dir_base, exist_ok=True)

    if copy_php:
        raise NotImplementedError("copy_php not implemented yet")
        copyIndexPHP(plot_dir_base)

    if plot_fmt:
        formats = [plot_fmt]

    paths = []
    for fmt in formats:
        plot_path_ = plot_path if plot_fmt else plot_path + f".{fmt}"
        paths.append(plot_path_)
        fig.savefig(plot_path_, **kwargs)
    logger.info(f"Plot saved in:" + "".join(["\n\t"+p for p in paths]))

        # show_url(plot_path)


# %%
