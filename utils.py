import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from utilsforecast.losses import mape as _mape


get_colors = lambda n: sns.husl_palette(n_colors=n)


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


def plot_diagnostics(data):
    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(2, 2, (1, 2))
    ax1.plot(data["ds"], data["resid"])
    ax1.set_title("Innovation Residuals")

    ax2 = fig.add_subplot(2, 2, 3)
    plot_acf(
        data["resid"].dropna(),
        ax=ax2,
        zero=False,
        bartlett_confint=False,
        auto_ylims=True,
    )
    ax2.set_title("ACF Plot")
    ax2.set_xlabel("lag[1]")

    ax3 = fig.add_subplot(2, 2, 4)
    ax3.hist(data["resid"], bins=20)
    ax3.set_title("Histogram")
    ax3.set_xlabel(".resid")
    ax3.set_ylabel("Count")

    plt.tight_layout()
    plt.show()


def mape(df, models, id_col="unique_id", target_col="y"):
    df_mape = _mape(df, models, id_col=id_col, target_col=target_col)
    df_mape.loc[:, df_mape.select_dtypes(include="number").columns] *= 100

    return df_mape
