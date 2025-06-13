import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
from statsmodels.graphics.tsaplots import plot_acf
from utilsforecast.losses import mape as _mape
from scipy import stats
from sklearn.metrics import r2_score


get_colors = lambda n: sns.husl_palette(n_colors=n)


def corrfunc(x, y, **kws):
    r, pvalue = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate(
        f"Corr: \n{r:.3f}{'***' if pvalue < 0.05 else ''}",
        xy=(0.5, 0.5),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=12,
    )


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


def adj_r2_score(y, y_hat, T, k):
    r2 = r2_score(y, y_hat)
    adj_r2 = 1 - (1 - r2) * (T - 1) / (T - k - 1)
    return adj_r2


def aic_score(y, y_hat, T, k):
    sse = np.sum((y - y_hat) ** 2)
    aic = T * np.log(sse / T) + 2 * (k + 2)
    return aic


def aicc_score(y, y_hat, T, k):
    aic = aic_score(y, y_hat, T, k)
    aicc = aic + (2 * (k + 2) * (k + 3)) / (T - k - 3)
    return aicc


def bic_score(y, y_hat, T, k):
    sse = np.sum((y - y_hat) ** 2)
    bic = T * np.log(sse / T) + (k + 2) * np.log(T)
    return bic


def cv_score(mf, df, model, target_col, n_windows, h):
    cv_predictions = mf.cross_validation(
        df, fitted=True, n_windows=n_windows, h=h, static_features=[]
    )
    cv_score = np.mean(
        np.sum((cv_predictions[target_col] - cv_predictions[model]) ** 2)
    )

    return cv_score


def print_regression_summary_from_model(model):
    X = model._X.values.astype(float)
    y = model._y
    residuals = model._residuals

    n, p = X.shape
    X_design = np.hstack([np.ones((n, 1)), X])
    df = n - p - 1

    res_summary = np.percentile(residuals, [0, 25, 50, 75, 100])
    print("#> Residuals:")
    print(f"#>     Min      1Q  Median      3Q     Max ")
    print(
        f"#> {res_summary[0]:7.4f} {res_summary[1]:7.4f} {res_summary[2]:7.4f} "
        f"{res_summary[3]:7.4f} {res_summary[4]:7.4f}\n"
    )

    coef = np.insert(model.coef_, 0, model.intercept_)
    rss = np.sum(residuals**2)
    mse = rss / df
    var_betas = mse * np.linalg.inv(X_design.T @ X_design).diagonal()
    se_betas = np.sqrt(var_betas)
    t_stats = coef / se_betas
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df))

    print("#> Coefficients:")
    print(
        f"#> {'':>13} {'Estimate':>9} {'Std. Error':>11} {'t value':>9} {'Pr(>|t|)':>9}"
    )
    names = ["(Intercept)"] + model._var_names
    for name, est, se, t, p_val in zip(names, coef, se_betas, t_stats, p_values):
        stars = (
            "***"
            if p_val < 0.001
            else (
                "**"
                if p_val < 0.01
                else "*" if p_val < 0.05 else "." if p_val < 0.1 else ""
            )
        )
        print(f"#> {name:>13} {est:9.4f} {se:11.4f} {t:9.2f} {p_val:9.3g} {stars}")
    print("---")
    print("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")

    r_squared = model.score(X, y)
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / df
    f_stat = (r_squared / p) / ((1 - r_squared) / df)
    f_pval = 1 - stats.f.cdf(f_stat, p, df)

    print(f"\nResidual standard error: {np.sqrt(mse):.3f} on {df} degrees of freedom")
    print(
        f"Multiple R-squared: {r_squared:.3f},   Adjusted R-squared: {adj_r_squared:.3f}"
    )
    print(f"F-statistic: {f_stat:.1f} on {df} DF, p-value: {f_pval:.3g}")


def plot_diagnostics_from_model(
    forecaster,
    model=None,
    n_lags=None,
    target_col="y",
    id_col="unique_id",
    time_col="ds",
):

    # Plot first model only if no model spec is given
    if model is None:
        model = list(forecaster.models.keys())[0]

    fitted_values = forecaster.forecast_fitted_values()
    insample_forecasts = fitted_values[model]
    residuals = fitted_values[target_col] - insample_forecasts

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(2, 2, (1, 2))
    ax1.plot(fitted_values[time_col], residuals, marker="o")
    ax1.set_title("Residuals")
    ax1.set_ylabel("Residuals")

    ax2 = fig.add_subplot(2, 2, 3)
    plot_acf(
        residuals.dropna(),
        ax=ax2,
        zero=False,
        lags=n_lags,
        bartlett_confint=False,
        auto_ylims=True,
    )
    ax2.set_xlabel("lag [1Q]")
    ax2.set_ylabel("ACF")

    ax3 = fig.add_subplot(2, 2, 4)
    ax3.hist(residuals, bins=20)
    ax3.set_title("Histogram")
    ax3.set_xlabel("Residuals")
    ax3.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()
