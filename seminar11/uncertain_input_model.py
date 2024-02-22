import numpy as np
import gpytorch
import torch

import sys

# sys.path.append("/app")
# sys.path.append("./Predictive-Validation-proper-horizon/tools")

from numpy.random import RandomState
from makey_glass import mackey_glass, MackeyGlassDataset
from gpr import GPRModel

# from scores import get_uct_metrics
import matplotlib.pyplot as plt
import numpy as np


class GPRUncInputs:
    def __init__(self, base_gp_model, starting_point: np.ndarray) -> None:
        """
        Implements iterative GP time series prediction with uncertain inputs:
        Prediction at an Uncertain Input for Gaussian Processes and Relevance Vector Machines Application to Multiple-Step Ahead Time-Series
        http://quinonero.net/Publications/QuiGirRas03.pdf
        :param base_gp_model: GP model that is used to produce posteriory based on input
        :param starting_point: The first L points of test time series, from which interative prediction process starts.
            We are certain about these points, i.e mean vector u=starting_point, S - zero matrix
        """
        self.base_gp_model = base_gp_model
        self.Lambda = np.diag(
            self.base_gp_model.model.covar_module.lengthscale.cpu().detach().numpy()[0]
            ** 2
        )
        self.Lambda_inv = np.diag(
            1
            / (
                base_gp_model.model.covar_module.lengthscale.cpu().detach().numpy()[0]
                ** 2
            )
        )
        self.u = starting_point
        self.L = self.u.shape[0]
        self.S = np.zeros((self.L, self.L))

        self.train_points = self.base_gp_model.model.train_inputs[0].numpy()
        self.train_targets = self.base_gp_model.model.train_targets.numpy()

        self.Sigma = (
            base_gp_model.model.covar_module(self.base_gp_model.model.train_inputs[0])
            .to_dense()
            .detach()
            .numpy()
        )
        # self.Sigma_inv = np.linalg.inv(self.Sigma)
        self.nose_estimate = self.base_gp_model.model.likelihood.noise.item()
        K = self.Sigma + self.nose_estimate * np.identity(self.train_points.shape[0])
        self.K_inv = np.linalg.inv(K + 10**-5 * np.identity(self.train_points.shape[0]))
        self.beta = self.K_inv @ self.train_targets

        self.random_input_part_size = 0  # part of input vector u, which is random.
        # at first whole vector u is deterministic

    @staticmethod
    def compute_quadratic_form(A, x):
        x_outer = x[..., :, np.newaxis] * x[..., np.newaxis, :]
        return np.sum(A[..., :, :] * x_outer, axis=(-1, -2))

    def compute_l(self):
        # see eqs. (33)
        l_norm = np.linalg.det(self.Lambda_inv @ self.S + np.identity(self.L)) ** -0.5
        s_l_inv = np.linalg.inv(self.S + self.Lambda)
        diff = self.u[np.newaxis, :] - self.train_points
        qw_form = self.compute_quadratic_form(s_l_inv, diff)
        l = l_norm * np.exp(-0.5 * qw_form)
        # u = torch.tensor(self.u, dtype=torch.float64)[np.newaxis,:] #torch.zeros(L, dtype=torch.float64)[np.newaxis,:]
        # train_points = self.base_gp_model.model.train_inputs[0]
        # k = gpr_model.model.covar_module(u,train_points).to_dense().detach().numpy()
        return l

    def compute_L_matrix(self):
        # see eqs. (37)
        L_norm = (
            np.linalg.det(2 * self.Lambda_inv @ self.S + np.identity(self.L)) ** -0.5
        )
        x_d = (
            self.train_points[np.newaxis, :, :] + self.train_points[:, np.newaxis, :]
        ) / 2
        x_minis = (
            self.train_points[np.newaxis, :, :] - self.train_points[:, np.newaxis, :]
        )
        u_minis_x_d = self.u[np.newaxis, np.newaxis, :] - x_d

        Lambda_plus_S_inv = np.linalg.inv(self.Lambda / 2 + self.S)
        Lambda_inv_half = self.Lambda_inv / 2

        return L_norm * np.exp(
            -0.5
            * (
                self.compute_quadratic_form(Lambda_plus_S_inv, u_minis_x_d)
                + self.compute_quadratic_form(Lambda_inv_half, x_minis)
            )
        )

    def compute_k(self):
        u = torch.tensor(self.u, dtype=torch.float64)[np.newaxis, :]
        train_points = torch.tensor(self.train_points, dtype=torch.float64)

        k = (
            self.base_gp_model.model.covar_module(u, train_points)
            .to_dense()
            .detach()
            .numpy()[0]
        )
        return k

    def predict_with_unc_inputs(self):
        # predicts next point mean and variance based on current uncertain input
        # see eqs. (30) for the mean and (35) for the variance
        # http://quinonero.net/Publications/QuiGirRas03.pdf

        # compute mean
        self.l = self.compute_l()
        m = np.dot(self.beta, self.l)

        # compute variance
        # 1. get GP variance
        y_pred_gp, v_gp = self.base_gp_model.predict(self.u[np.newaxis, :])
        v_gp = v_gp[0] ** 2
        # 2. get anxilary marixes
        self.k = self.compute_k()
        self.L_matrix = self.compute_L_matrix()
        k_outer = self.k[np.newaxis, :] * self.k[:, np.newaxis]
        l_outer = self.l[np.newaxis, :] * self.l[:, np.newaxis]
        beta_outer = self.beta[np.newaxis, :] * self.beta[:, np.newaxis]
        v = (
            v_gp
            + np.trace(self.K_inv @ (k_outer - self.L_matrix))
            + np.trace(beta_outer @ (self.L_matrix - l_outer))
        )

        # update matrix S and mean vector
        self.update_input_distr(v, m)

        return m, v**0.5

    def update_input_distr(self, v_pred, m_pred):
        # update covariance matrix S and mean vector u after new predicted point will be added
        # see eqs. (53)
        # at first iterations not whole vector u is random
        # thus we want to compute variances between new predicted point y_{T+K} and random part
        # of vector u

        u_new = np.zeros_like(self.u)
        S_new = np.zeros_like(self.S)

        # update vector u
        u_new[0] = m_pred
        u_new[1:] = self.u[:-1]

        # update matrix S
        S_Lambda_inv = np.linalg.inv(self.S + self.Lambda)
        lambda_train = ((self.S @ S_Lambda_inv) @ self.train_points.T).T
        c_j = lambda_train + ((self.Lambda @ S_Lambda_inv) @ self.u)[np.newaxis, :]
        cov = np.sum(
            self.beta[:, np.newaxis]
            * self.l[:, np.newaxis]
            * (c_j - self.u[np.newaxis, :]),
            axis=0,
        )

        S_new[0, 0] = v_pred + self.nose_estimate
        S_new[1:, 1:] = self.S[:-1, :-1]
        S_new[0, 1:] = cov[:-1]
        S_new[1:, 0] = cov[:-1]
        self.S = S_new
        self.u = u_new
        self.random_input_part_size += 1
        return


if __name__ == "__main__":
    L = 50
    nose_std = (0.001) ** 0.5
    seed = 7
    makey_train_ds = MackeyGlassDataset(
        length=8000, size=100, L=L, seed=seed, nose_std=nose_std
    )
    # create test point

    rs = RandomState(seed)
    n = 170
    x0 = 0.5 + 0.05 * (-1 + 2 * rs.random(n))
    x = mackey_glass(length=100 + L, tau=17.0, sample=1.0, n=n, x0=x0)
    test_points = (x - makey_train_ds.mean) / makey_train_ds.std
    test_points += rs.normal(0, nose_std, size=len(test_points))

    gpr_model = GPRModel(normalize_data=False, training_iter=500, use_scale=False)

    x_train = makey_train_ds.train_set_x
    y_train = makey_train_ds.train_set_y

    # fit
    gpr_model.fit(np.flip(x_train, 1).copy(), y_train)

    # test unc model
    model = GPRUncInputs(gpr_model, test_points[:L][::-1].copy())
    predictions = []
    stds = []
    for i in range(len(test_points) - L):
        next_x, next_std = model.predict_with_unc_inputs()
        predictions.append(next_x)
        stds.append(next_std)

    predictions = np.array(predictions)
    stds = np.array(stds)
    val_low = predictions - 2 * stds
    val_high = predictions + 2 * stds

    # predict gpr
    predictions_gpr = []
    stds_gpr = []
    x_test = test_points[:L][::-1].tolist()
    for i in range(len(test_points) - L):
        next_x, next_std = gpr_model.predict(np.array(x_test)[np.newaxis, :])
        predictions_gpr.append(next_x[0])
        stds_gpr.append(next_std[0])
        x_test = [next_x[0]] + x_test[:-1]

    predictions_gpr = np.array(predictions_gpr)
    stds_gpr = np.array(stds_gpr)
    val_low_gpr = predictions_gpr - 2 * stds_gpr
    val_high_gpr = predictions_gpr + 2 * stds_gpr

    plt.figure(figsize=(8, 5), dpi=300)
    time_future = np.arange(len(test_points) - L)
    plt.plot(
        time_future,
        test_points[L:],
        label="true",
        linewidth=2,
        linestyle="-",
        marker="o",
        markersize=2,
    )
    plt.plot(
        time_future,
        predictions_gpr,
        label="GPR",
        linewidth=2,
        linestyle="--",
        marker="o",
        markersize=2,
    )
    plt.plot(
        time_future,
        predictions,
        label="UncInputGPR",
        linewidth=2,
        linestyle="--",
        marker="o",
        markersize=2,
    )

    plt.fill_between(time_future, val_low, val_high, alpha=0.2)
    plt.fill_between(time_future, val_low_gpr, val_high_gpr, alpha=0.2)

    plt.xticks(fontsize=14, rotation=60)
    plt.yticks(fontsize=14)
    plt.xlabel("time step", fontsize=18)
    plt.ylabel("target", fontsize=18)
    plt.legend(fontsize=14)
    plt.savefig("./test.png")
    # print([model.Lambda[i][i] for i in range(L)])
    # plt.savefig("/app/outputs/unc_inputs/test.png")
    # print("/app/outputs/unc_inputs/test.png")
    # d = get_uct_metrics(test_points[L:], predictions, stds)
    # d_gpr = get_uct_metrics(test_points[L:], predictions_gpr, stds_gpr)
