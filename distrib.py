import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from joblib import Parallel, delayed
from bridge import BRidge, generate_regression, function_trigonometric

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# From https://stackoverflow.com/questions/25812255
def add_headers(
    fig,
    *,
    row_headers=None,
    col_headers=None,
    row_pad=8,
    col_pad=8,
    rotate_row_headers=True,
    **text_kwargs
):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        if (row_headers is not None) and sbs.is_last_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(1, 0.5),
                xytext=(row_pad, 0),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="left",
                va="center",
                rotation=rotate_row_headers * 270,
                **text_kwargs,
            )


def compute_distrib_of_w_grid(function, n_list, r_list, n_samples, qlip_q=0.999):

    def generate_optimal_weight(n, r):
        y, Psi, _ = generate_regression(n, function)
        w = BRidge(r=r, fit_intercept=False).fit(Psi, y).get_w()
        return w

    sns.set_theme(style="whitegrid", palette="muted")

    fig, axes = plt.subplots(
        len(n_list), len(r_list), figsize=(5 * len(r_list), 4 * len(n_list)),
        sharex=True, sharey=False
    )

    var_table = pd.DataFrame(index=n_list, columns=r_list)
    var_table_n = pd.DataFrame(index=n_list, columns=r_list)

    for i_r, r in enumerate(r_list):
        for i_n, n in enumerate(n_list):
            weights = np.array(
                Parallel(n_jobs=-1)(delayed(generate_optimal_weight)(n, r) for _ in range(n_samples))
            )
            
            weights = weights.clip(0, np.quantile(weights, qlip_q))
            sns.histplot(weights.flatten(), kde=True, stat="density", ax=axes[i_n, i_r], linewidth=1.5, bins=20)

            var_table.at[n, r] = np.var(weights.flatten())
            var_table_n.at[n, r] = np.var(weights.flatten()) * n

            axes[i_n, i_r].set_xlabel(r'$\widetilde{w}$', fontsize=12, labelpad=8)
            axes[i_n, i_r].set_ylabel("density", fontsize=12, labelpad=8)
    
    font_kwargs = dict(fontsize="large")
    add_headers(fig, row_headers=[f"{r'$n$'} = {n}" for n in n_list] , col_headers=[f"{r'$r$'} = {r}" for r in r_list], **font_kwargs)
    plt.tight_layout()

    filename = f"figures/plot_distib_grid_{n_samples}.png"
    plt.savefig(filename, dpi=600)

    latex_table = var_table.to_latex(float_format="%.2f", caption="Variance of weights for different $n$ and $r$", label="tab:std_weights")
    
    with open(f"figures/var_table_{n_samples}.tex", "w") as f:
        f.write(latex_table)

    latex_table = var_table_n.to_latex(float_format="%.2f", caption="Variance of weights times $n$ for different $n$ and $r$", label="tab:std_weights")
    
    with open(f"figures/var_table_{n_samples}_times_n.tex", "w") as f:
        f.write(latex_table)


if __name__ == "__main__":
    np.random.seed(42)

    size = 10
    a = np.random.normal(0, 0.16, size=size)
    b = np.random.normal(0, 0.16, size=size)
    function = lambda x: function_trigonometric(x, a, b, size)
    
    compute_distrib_of_w_grid(function=function, n_list=[50, 250, 1000, 4000], r_list=[1, 6, 12], n_samples=10_000)
