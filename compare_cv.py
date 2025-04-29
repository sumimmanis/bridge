import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib.gridspec as gridspec

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from tqdm import tqdm
from bridge import BRidge, Tikhonov, generate_regression, function_trigonometric

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


cv = 5


def generate_train_test_families(n, m, s, seed):
    np.random.seed()
    size = 10
    a = np.random.normal(0, 0.16, size=size)
    b = np.random.normal(0, 0.16, size=size)

    function = lambda x: function_trigonometric(x, a, b, size)

    _, X_train, _ = generate_regression(50_000, function)
    ss = StandardScaler().fit(X_train)

    y_test, X_test, _ = generate_regression(n * s, function)
    X_test = ss.transform(X_test)

    train_sets = []
    for i in range(m):
        y_train, X_train, _ = generate_regression(n, function)
        X_train = ss.transform(X_train)
        train_sets.append((X_train, y_train))

    return train_sets, X_test, y_test


def generate_train_test_california(n, m, s):
    np.random.seed(23)

    X, y = fetch_california_housing(return_X_y=True)

    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=s, random_state=42
    )

    ss = StandardScaler()
    X_rest = ss.fit_transform(X_rest)
    X_test = ss.transform(X_test)

    train_sets = []
    cnt=0
    for i in range(m):
        if i == 0 or (cnt+1) * n > len(perm):
            perm = np.random.permutation(len(X_rest))
            cnt = 0

        idx = perm[cnt * n:(cnt+1) * n]
        cnt += 1

        X_train = X_rest[idx]
        y_train = y_rest[idx]

        train_sets.append((X_train, y_train))

    return train_sets, X_test, y_test


def make_G_sobolev_trigonometric(size=10, smoothness=2):
    freqs = np.arange(1, size + 1)
    penalties = freqs**smoothness
    
    G = np.diag(np.hstack([penalties, penalties]))
    return G


def compute_oracle(train_sets, X_test, y_test, fit_intercept, w_values, G=None):
    mean_mses = []

    for w in tqdm(w_values):
        mean_mses.append(
            np.mean([mean_squared_error(y_test, Tikhonov(G=G, fit_intercept=fit_intercept, w=w)
                    .fit(X_train, y_train).predict(X_test)) for X_train, y_train in train_sets])
        )
            
    return w_values[np.argmin(mean_mses)]


def compute_function_oracle(X_train, y_train, X_test, y_test, fit_intercept, w_values, G=None):
    mean_mses = []

    for w in w_values:
        model = Tikhonov(G=G, fit_intercept=fit_intercept, w=w).fit(X_train, y_train)
        mean_mses.append(mean_squared_error(y_test, model.predict(X_test)))

    return w_values[np.argmin(mean_mses)]


###########################################################################################################


def run_experiment(oracle_w, train_sets, X_test, y_test, fit_intercept, w_values, r_values, G=None):
    BRidge_mses = []
    tikhonov_mses = []
    BRidge_w = []
    tikhonov_w = []
    w_errors_BRidge = []
    w_errors_tikhonov = []
    w_errors_oracle_BRidge = []
    w_errors_oracle_tikhonov = []
    selected_r = []


    Tikhonov_ = GridSearchCV(
        Tikhonov(G=G, fit_intercept=fit_intercept), 
        {'w': w_values},
        cv=cv,
        scoring='neg_mean_squared_error'
    )

    BRidge_ = GridSearchCV(
        BRidge(G=G, fit_intercept=fit_intercept), 
        {'r': r_values},
        cv=cv,
        scoring='neg_mean_squared_error'
    )

    for X_train, y_train in tqdm(train_sets):
        current_oracle_w = compute_function_oracle(X_train, y_train, X_test, y_test, fit_intercept, w_values, G)

        # BRidge
        bridge = BRidge_.fit(X_train, y_train)
        y_pred_BRidge = bridge.predict(X_test)

        BRidge_mses.append(mean_squared_error(y_test, y_pred_BRidge))
        BRidge_w.append(bridge.best_estimator_.get_w())
        w_errors_BRidge.append(np.abs(bridge.best_estimator_.get_w() - current_oracle_w))
        w_errors_oracle_BRidge.append(np.abs(bridge.best_estimator_.get_w() - oracle_w))
        selected_r.append(bridge.best_estimator_.r)

        # Tikhonov
        tikhonov = Tikhonov_.fit(X_train, y_train)
        y_pred_tikhonov = tikhonov.predict(X_test)

        tikhonov_mses.append(mean_squared_error(y_test, y_pred_tikhonov))
        tikhonov_w.append(tikhonov.best_params_['w'])
        w_errors_tikhonov.append(np.abs(tikhonov.best_params_['w'] - current_oracle_w))
        w_errors_oracle_tikhonov.append(np.abs(tikhonov.best_params_['w'] - oracle_w))

    results = {
            "BRidge": {
                "Test MSE": BRidge_mses,
                "Selected w": BRidge_w,
                "w Error current": w_errors_BRidge,
                "w Error oracle": w_errors_oracle_BRidge,
                "selected_r": selected_r
            },
            "Tikhonov": {
                "Test MSE": tikhonov_mses,
                "Selected w": tikhonov_w,
                "w Error current": w_errors_tikhonov,
                "w Error oracle": w_errors_oracle_tikhonov,
            },
            "Oracle w": oracle_w,
        }
    return results


###########################################################################################################


def plot_helper(df_mse, df_w, df_error, r_values, oracle_w, dataset_name):
    boxargs = dict(showmeans=True)
    
    fig = plt.figure(figsize=(22, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 0.15, 1, 1])
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    # Plot (A): Test MSE
    sns.boxplot(x="Method", y="Test MSE", data=df_mse, ax=axes[0], **boxargs)
    mse_bridge = df_mse[df_mse['Method'] == 'BRidge']['Test MSE'].mean()
    mse_ridge = df_mse[df_mse['Method'] == 'Tikhonov']['Test MSE'].mean()

    axes[0].set_title(f"(A) Mean squared error of BRidge and Ridge estimators\n"
                      f"mean    BRidge: {mse_bridge:.4f},  Ridge: {mse_ridge:.4f}")
    axes[0].set_xlabel("")
    axes[0].set_ylabel(r"$(f(X) - Y)^2$")

    # Plot: Histogram of selected r
    sns.histplot(y=r_values, ax=axes[1], bins=len(set(r_values)), stat="count")
    axes[1].set_title("")
    axes[1].set_xlabel("Count")
    axes[1].set_ylabel(r"$\widetilde{r}$")

    # Plot (B): Selected w
    sns.boxplot(x="Method", y="w", data=df_w, ax=axes[2], **boxargs)
    w_bridge = np.median(df_w[df_w['Method'] == 'BRidge']['w'])
    w_ridge = np.median(df_w[df_w['Method'] == 'Tikhonov']['w'])

    axes[2].axhline(oracle_w, color="red", linestyle="--", label=r"$w^*$")
    axes[2].set_title(f"(B) Cross-validated regularization parameters versus oracle\n"
                      f"|oracle - median|    BRidge {abs(w_bridge - oracle_w):.4f},  "
                      f"Ridge {abs(w_ridge - oracle_w):.4f}")
    axes[2].set_xlabel("")
    axes[2].set_ylabel(r"$\widetilde{w}_X$")
    axes[2].legend()

    # Plot (C): Squared error
    sns.boxplot(x="Method", y="Squared Error", data=df_error, ax=axes[3], **boxargs)
    error_bridge = df_error[df_error['Method'] == 'BRidge']['Squared Error'].mean()
    error_ridge = df_error[df_error['Method'] == 'Tikhonov']['Squared Error'].mean()

    axes[3].set_title(f"(C) Absolute deviation between cross-validated and oracle parameters\n"
                      f"mean    BRidge: {error_bridge:.4f},  Ridge: {error_ridge:.4f}")
    axes[3].set_xlabel("")
    axes[3].set_ylabel(r"$|w^\circ_X - \widetilde{w}_X|$")

    plt.tight_layout()
    plt.savefig(f"figures/{dataset_name}_boxplot_r_distib.png", dpi=600, bbox_inches="tight")
    plt.close()

def plot_results(results, dataset_name):
    df_mse = pd.DataFrame({
        "Method": ["BRidge"] * len(results["BRidge"]["Test MSE"]) +
                  ["Tikhonov"] * len(results["Tikhonov"]["Test MSE"]),
        "Test MSE": results["BRidge"]["Test MSE"] + results["Tikhonov"]["Test MSE"]
    })

    df_w = pd.DataFrame({
        "Method": ["BRidge"] * len(results["BRidge"]["Selected w"]) +
                  ["Tikhonov"] * len(results["Tikhonov"]["Selected w"]),
        "w": results["BRidge"]["Selected w"] + results["Tikhonov"]["Selected w"]
    })

    df_error = pd.DataFrame({
        "Method": ["BRidge"] * len(results["BRidge"]["w Error current"]) +
                  ["Tikhonov"] * len(results["Tikhonov"]["w Error current"]),
        "Squared Error": results["BRidge"]["w Error current"] + results["Tikhonov"]["w Error current"]
    })

    plot_helper(df_mse, df_w, df_error, results["BRidge"]['selected_r'], results["Oracle w"], dataset_name)


if __name__ == "__main__":

    m = 100
    s = 100

    seed = 42

    datasets = {
        'california_1000': (generate_train_test_california(n=1000, m=50, s=5000), True, np.linspace(0.00001, 1000, 500), np.linspace(0.00001, 1000, 500)), #size = 20640
        'california_2000': (generate_train_test_california(n=2000, m=50, s=5000), True, np.linspace(0.00001, 1000, 500), np.linspace(0.00001, 1000, 500)), #size = 20640
        'california_4000': (generate_train_test_california(n=4000, m=50, s=5000), True, np.linspace(0.00001, 1000, 500), np.linspace(0.00001, 1000, 500)), #size = 20640
        'function_trigonometric_250': (generate_train_test_families(n=250, m=m, s=s, seed=seed), False, np.linspace(1, 500, 500), np.linspace(1, 2000, 500)),
        'function_trigonometric_1000': (generate_train_test_families(n=1000, m=m, s=s, seed=seed), False, np.linspace(1, 300, 500), np.linspace(1, 2000, 500)),
        }

    for dataset_name, ((train_sets, X_test, y_test), fit_intercept, w_values, r_values), in datasets.items():
        oracle_w = compute_oracle(train_sets, X_test, y_test, fit_intercept, w_values)
        results = run_experiment(oracle_w, train_sets, X_test, y_test, fit_intercept, w_values, r_values)

        pd.DataFrame(results).to_csv(f'saved_data/{dataset_name}.csv', index=True)

        plot_results(results, dataset_name)

    datasets = {
        'function_trigonometric_500_sobolev': (generate_train_test_families(n=500, m=m, s=s, seed=seed), False, np.linspace(0.00001, 1, 500), np.linspace(0.00001, 1, 500)),
        'function_trigonometric_2000_sobolev': (generate_train_test_families(n=2000, m=m, s=s, seed=seed), False, np.linspace(0.00001, 1, 500), np.linspace(0.00001, 1, 500)),
        }

    for dataset_name, ((train_sets, X_test, y_test), fit_intercept, w_values, r_values), in datasets.items():
        oracle_w = compute_oracle(train_sets, X_test, y_test, fit_intercept, w_values, G=make_G_sobolev_trigonometric())
        results = run_experiment(oracle_w, train_sets, X_test, y_test, fit_intercept, w_values, r_values, G=make_G_sobolev_trigonometric())

        pd.DataFrame(results).to_csv(f'saved_data/{dataset_name}.csv', index=True)

        plot_results(results, dataset_name)

    seed = 16
    
    datasets = {
        'function_trigonometric_250_2': (generate_train_test_families(n=250, m=m, s=s, seed=seed), False, np.linspace(1, 500, 500), np.linspace(1, 2000, 500)),
        'function_trigonometric_1000_2': (generate_train_test_families(n=1000, m=m, s=s, seed=seed), False, np.linspace(1, 300, 500), np.linspace(1, 2000, 500)),
        }

    for dataset_name, ((train_sets, X_test, y_test), fit_intercept, w_values, r_values), in datasets.items():
        oracle_w = compute_oracle(train_sets, X_test, y_test, fit_intercept, w_values)
        results = run_experiment(oracle_w, train_sets, X_test, y_test, fit_intercept, w_values, r_values)

        pd.DataFrame(results).to_csv(f'saved_data/{dataset_name}.csv', index=True)

        plot_results(results, dataset_name)

    datasets = {
        'function_trigonometric_500_sobolev_2': (generate_train_test_families(n=500, m=m, s=s, seed=seed), False, np.linspace(0.00001, 1, 500), np.linspace(0.00001, 1, 500)),
        'function_trigonometric_2000_sobolev_2': (generate_train_test_families(n=2000, m=m, s=s, seed=seed), False, np.linspace(0.00001, 1, 500), np.linspace(0.00001, 1, 500)),
        }

    for dataset_name, ((train_sets, X_test, y_test), fit_intercept, w_values, r_values), in datasets.items():
        oracle_w = compute_oracle(train_sets, X_test, y_test, fit_intercept, w_values, G=make_G_sobolev_trigonometric())
        results = run_experiment(oracle_w, train_sets, X_test, y_test, fit_intercept, w_values, r_values, G=make_G_sobolev_trigonometric())

        pd.DataFrame(results).to_csv(f'saved_data/{dataset_name}.csv', index=True)

        plot_results(results, dataset_name)
