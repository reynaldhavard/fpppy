import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


get_colors = lambda n: sns.husl_palette(n_colors=n)


def plot_series(
    df,
    id_col,
    time_col,
    target_col,
    title,
    xlabel,
    ylabel,
    ids=None,
    max_insample_length=None,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    df_grouped = df.groupby(id_col)
    if ids is not None:
        group_keys = ids
    else:
        group_keys = df_grouped.groups.keys()
    colors = get_colors(n=len(group_keys))
    for i, key in enumerate(group_keys):
        group = df_grouped.get_group(key)
        if ids is not None and key not in ids:
            continue
        if max_insample_length is not None:
            group = group.tail(max_insample_length)
        ax.plot(group[time_col], group[target_col], label=key, color=colors[i])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ids is not None and len(ids) > 1:
        ax.legend(title=id_col, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def lag_plot(df, target_col, lags, period_name, labels, nrows, ncols):
    tmp_df = df.copy()
    for lag in range(1, lags + 1):
        tmp_df[f"lag_{lag}"] = tmp_df[target_col].shift(lag)
    lims = [
        np.min(
            [tmp_df[f"lag_{i}"].min() for i in range(1, lags + 1)]
            + [tmp_df[target_col].min()]
        ),
        np.max(
            [tmp_df[f"lag_{i}"].max() for i in range(1, lags + 1)]
            + [tmp_df[target_col].max()]
        ),
    ]
    colors = get_colors(n=df[period_name].nunique())
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
    for ax, lag in zip(axes.flatten(), range(1, lags + 1)):
        sns.scatterplot(
            data=tmp_df,
            x=f"lag_{lag}",
            y=target_col,
            ax=ax,
            hue=period_name,
            palette=colors,
            legend=False,
        )
        ax.plot(lims, lims, "grey", linestyle="--", linewidth=1)
        ax.set_title(f"lag_{lag}")

    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
    fig.legend(
        handles,
        labels,
        title="Season",
        loc="center right",
        fontsize=10,
        title_fontsize=12,
        bbox_to_anchor=(1.1, 0.5),
    )
    fig.supxlabel(f"lag({target_col}, k)", y=0.05, fontsize=12)
    fig.supylabel(target_col, x=0.07, fontsize=12)
    plt.tight_layout()
    plt.show()
