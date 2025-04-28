import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


def function_trigonometric(x, a, b, size):        
    inner = 2 * np.pi * np.arange(1, size + 1).reshape(1, -1) * x.reshape(-1, 1)
    Psi = np.hstack([np.sin(inner), np.cos(inner)])
    y_true = np.sum(a * np.sin(inner), axis=1) + np.sum(b * np.cos(inner), axis=1)
    return Psi, y_true


def generate_regression(n, function, x_l=0, x_r=1):
    x = np.random.uniform(x_l, x_r, n)
    Psi, y_true = function(x)
    y = y_true + np.random.normal(0, 1, n)
    return y, Psi, y_true


def get_theta_opt(y, Psi, G, w):
    A = w * G.T @ G + Psi.T @ Psi
    b = Psi.T @ y
    theta = np.linalg.solve(A, b)
    return theta


def optimize_theta_w(y, Psi, G, r, max_iter):
    w = 1.0
    for i in range(max_iter):
        w_old = w
        theta = get_theta_opt(y, Psi, G, w)
        w = np.sqrt(2 * r) / np.linalg.norm(G @ theta)
        if np.isclose(w_old, w):
            break
    else:
        print(f"Warning: Optimization did not converge within the maximum number of iterations for r={r}")
    theta = get_theta_opt(y, Psi, G, w)   
    return theta, w


class BRidge(BaseEstimator, RegressorMixin):
    def __init__(self, G=None, r=1.0, fit_intercept=True, max_iter=100):
        self.G = G
        self.r = r
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.w = None
    
    def fit(self, Psi, y):
        if self.G is None:
            self.G = np.eye(Psi.shape[1])

        if self.fit_intercept:
            Psi_mean = np.mean(Psi, axis=0)
            y_mean = np.mean(y)

            self.theta_, self.w = optimize_theta_w(
                y - y_mean, Psi - Psi_mean, self.G, self.r, self.max_iter
            )
            self.intercept_ = y_mean - Psi_mean @ self.theta_
        else:
            self.theta_, self.w = optimize_theta_w(
                y, Psi, self.G, self.r, self.max_iter
            )
            self.intercept_ = 0

        return self

    def predict(self, Psi):
        return Psi @ self.theta_ + self.intercept_
    
    def get_w(self):
        return self.w


class Tikhonov(BaseEstimator, RegressorMixin):
    def __init__(self, G=None, w=1.0, fit_intercept=True):
        self.G = G
        self.w = w
        self.fit_intercept = fit_intercept
    
    def fit(self, Psi, y):
        if self.G is None:
            self.G = np.eye(Psi.shape[1])

        if self.fit_intercept:
            Psi_mean = np.mean(Psi, axis=0)
            y_mean = np.mean(y)

            self.theta_ = get_theta_opt(
                y - y_mean, Psi - Psi_mean, self.G, self.w
            )
            self.intercept_ = y_mean - Psi_mean @ self.theta_
        else:
            self.theta_ = get_theta_opt(
                y, Psi, self.G, self.w
            )
            self.intercept_ = 0

        return self

    def predict(self, Psi):
        return Psi @ self.theta_ + self.intercept_
