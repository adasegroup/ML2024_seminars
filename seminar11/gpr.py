from typing import Tuple, Optional, Any, List
import warnings
from abc import abstractmethod, ABC

import numpy as np
import scipy
import torch
import gpytorch
from gpytorch.kernels import LinearKernel, RBFKernel, ScaleKernel


class BaseModel(ABC):
    """Abstract BaseModel wrapper."""

    def __init__(
        self,
        model: Any,
        model_upper_bound: Optional[Any] = None,
        model_lower_bound: Optional[Any] = None,
    ) -> None:
        """Abstract base class for model's benchmarking.

        :param model: Any predictive model
        :param model_upper_bound: Any model predicting upper bound of the confident interval
        :param model_lower_bound: Any model predicting lower bound of the confident interval
        """
        self.model = model
        self.model_upper_bound = model_upper_bound
        self.model_lower_bound = model_lower_bound

        self.preds = None
        self.std = None
        self.preds_low = None
        self.preds_high = None

    @abstractmethod
    def fit(
        self, x_history: np.ndarray, y_history: np.ndarray, *args, **kwargs
    ) -> None:
        """Fit the main model.

        :param x_history: np.ndarray points history
        :param y_history: np.ndarray value history
        """
        raise NotImplementedError("fit method not implemented")

    @abstractmethod
    def predict(self, x_future: np.ndarray, *args, **kwargs) -> None:
        """Predict values for main model.

        :param x_future: np.ndarray future points
        """
        raise NotImplementedError("predict method not implemented")

    @abstractmethod
    def fit_bound_models(
        self, x_history: np.ndarray, y_history: np.ndarray, *args, **kwargs
    ) -> None:
        """Fit both bound models.

        :param x_history: np.ndarray points history
        :param y_history: np.ndarray value history
        """
        raise NotImplementedError("fit_bound_models method not implemented")

    @abstractmethod
    def predict_bound_models(self, x_history: np.ndarray, *args, **kwargs) -> None:
        """Predict values for both upper and lower confident interval's values.

        :param x_history: np.ndarray future points
        """
        raise NotImplementedError("predict_bound_models method not implemented")

    def fit_all(self, x_history: np.ndarray, y_history: np.ndarray) -> None:
        """Fit all models.

        :param x_history: np.ndarray points history
        :param y_history: np.ndarray value history
        """
        if self.model is not None:
            self.fit(x_history, y_history)
        if self.model_lower_bound is not None and self.model_upper_bound is not None:
            self.fit_bound_models(x_history, y_history)

    def predict_all(self, x_future: np.ndarray, y_future: np.ndarray) -> None:
        """Predict values and confident interval.

        :param x_future: np.ndarray future points
        """
        if self.model is not None:
            self.predict(x_future, y_future)
        if self.model_lower_bound is not None and self.model_upper_bound is not None:
            self.predict_bound_models(x_future)

    def get_preds(self) -> Optional[np.ndarray]:
        """Getter for predictions.

        :return: predictions
        """
        return self.preds

    def get_std(self) -> Optional[np.ndarray]:
        """Getter for std predictions.

        :return: std
        """
        return self.std

    def get_low_high_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Getter for interval bounds' predictions.

        :return: bounds
        """
        return self.preds_low, self.preds_high

    def stds_to_conf_interval(self, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate confidence interval using std

        :param alpha: confidence level
        :return: bounds of confidence interval
        """
        z_conf_interval = abs(scipy.stats.norm.ppf(alpha / 2))
        width = z_conf_interval * self.std / np.sqrt(len(self.preds))
        return self.preds - width, self.preds + width

    def conf_interval_to_stds(self, alpha: float) -> np.ndarray:
        """Calculate std using confidence interval

        :param alpha: confidence level
        :return: stds
        """
        width = (self.preds_high - self.preds_low) / 2
        return width * np.sqrt(len(self.preds)) * abs(scipy.stats.norm.ppf(alpha / 2))


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        num_features = train_x.shape[1]
        self.base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=num_features)
        # self.covar_module = gpytorch.kernels.MultitaskKernel(
        #     gpytorch.kernels.ScaleKernel(self.base_kernel),
        #     num_tasks=num_tasks,
        #     rank=num_tasks,
        # )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            self.base_kernel, num_tasks=num_tasks, rank=num_tasks
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


# --------------------------------------------------------------------------------- #
#                     GPR model                                                     #
# --------------------------------------------------------------------------------- #


class GeneralKernelModel(gpytorch.models.ExactGP):
    """Class implemented GP model with GeneralKernel."""

    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        ard_num_dims,
        kernel,
        continuous_features_idx,
        categorical_features_idx,
        use_scale=True,
    ):
        super(GeneralKernelModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if kernel == "linear":
            self.base_kernel = LinearKernel()
        elif kernel == "RBF":
            self.base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
        elif kernel == "periodic":
            self.base_kernel = gpytorch.kernels.PeriodicKernel(
                ard_num_dims=ard_num_dims
            )
        elif kernel == "linear_with_cat_kernel":
            self.base_kernel = LinearKernel(
                active_dims=torch.tensor(continuous_features_idx)
            )
            for cat_feature_idx in categorical_features_idx:
                self.base_kernel = self.base_kernel * RBFKernel(
                    active_dims=torch.tensor([cat_feature_idx])
                )

        else:
            raise ValueError("Unknown kernel type")

        if use_scale:
            self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
        else:
            self.covar_module = self.base_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def normalize_inputs(x_history, y_history, categorical_features_idx, eps):
    if categorical_features_idx is not None:
        continuous_features_idx = np.array(
            list(
                set(list(np.arange(x_history.shape[1]))).difference(
                    set(categorical_features_idx)
                )
            )
        )
    else:
        continuous_features_idx = np.arange(x_history.shape[1])
    # normatize data

    x_history_mean = np.mean(x_history[:, continuous_features_idx], axis=0)
    x_history_std = np.std(x_history[:, continuous_features_idx], axis=0)
    x_history_std = np.clip(x_history_std, eps, np.inf)

    x_history_norm = np.copy(x_history)
    x_history_norm[:, continuous_features_idx] = (
        x_history[:, continuous_features_idx] - x_history_mean
    ) / x_history_std

    # normalize values to have zero mean and 1 variance
    y_history_mean = np.mean(y_history)
    y_history_std = np.std(y_history)
    y_history_std = np.clip(y_history_std, eps, np.inf)

    y_history_norm = (y_history - y_history_mean) / y_history_std

    return (
        x_history_norm,
        y_history_norm,
        categorical_features_idx,
        continuous_features_idx,
        x_history_mean,
        x_history_std,
        y_history_mean,
        y_history_std,
    )


class GPRModel(BaseModel):
    """GPRModel wrapper."""

    def __init__(
        self,
        categorical_features_idx=None,
        train_variance=None,
        learn_additional_noise=False,
        test_variance=None,
        alpha: float = 0.05,
        training_iter: int = 200,
        kernel="RBF",
        normalize_data=True,
        use_scale=True,
    ) -> None:
        """Initialize GPRModel.

        :param model: not None placeholder
        :param alpha: confidence level
        """
        super().__init__(42)
        self.train_variance = train_variance
        self.learn_additional_noise = learn_additional_noise
        self.test_variance = test_variance
        self.alpha = alpha
        self.categorical_features_idx = categorical_features_idx
        self.training_iter = training_iter
        self.kernel = kernel
        self.normalize_data = normalize_data
        self.eps = 10**-4
        self.use_scale = use_scale

    def fit(
        self, x_history: np.ndarray, y_history: np.ndarray, *args, **kwargs
    ) -> None:
        """Fit the model.

        :param x_history: np.ndarray points history
        :param y_history: np.ndarray value history
        """

        if self.categorical_features_idx is None:
            self.continuous_features_idx = np.arange(x_history.shape[1])
            self.categorical_features_idx = np.array([])
        else:
            self.continuous_features_idx = np.array(
                list(
                    set(list(np.arange(x_history.shape[1]))).difference(
                        set(self.categorical_features_idx)
                    )
                )
            )
            self.categorical_features_idx = np.array(self.categorical_features_idx)

        # normatize data
        if self.normalize_data:
            (
                x_history,
                y_history,
                self.categorical_features_idx,
                self.continuous_features_idx,
                self.x_history_mean,
                self.x_history_std,
                self.y_history_mean,
                self.y_history_std,
            ) = normalize_inputs(
                x_history, y_history, self.categorical_features_idx, self.eps
            )
        # convert to torch tensors
        x_history = torch.tensor(x_history)
        y_history = torch.tensor(y_history)

        # create GPR model
        if self.train_variance is not None:
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=self.train_variance,
                learn_additional_noise=self.learn_additional_noise,
            )
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()

        num_features = x_history.shape[1]
        model = GeneralKernelModel(
            x_history,
            y_history,
            likelihood,
            ard_num_dims=num_features,
            kernel=self.kernel,
            continuous_features_idx=self.continuous_features_idx,
            categorical_features_idx=self.categorical_features_idx,
            use_scale=self.use_scale,
        )

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        # Includes GaussianLikelihood parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for _ in range(self.training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(x_history)
            # Calc loss and backpropagate gradients
            loss = -mll(output, y_history)
            loss.backward()
            optimizer.step()
        self.model = model
        self.likelihood = likelihood

    def fit_bound_models(
        self, x_history: np.ndarray, y_history: np.ndarray, *args, **kwargs
    ) -> None:
        """Fit both bound models.

        :param x_history: np.ndarray points history
        :param y_history: np.ndarray value history
        """
        warnings.warn("Not applicable for the GP model.")

    def predict(self, x_future: np.ndarray, *args, **kwargs) -> None:
        """Predict values for GPR model.

        :param x_future: np.ndarray points
        """
        x_future_test = np.copy(x_future)
        if self.normalize_data:
            x_future_test[:, self.continuous_features_idx] = (
                x_future[:, self.continuous_features_idx] - self.x_history_mean
            ) / self.x_history_std

        x_future_test = torch.tensor(x_future_test)

        self.model.eval()
        self.likelihood.eval()

        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if self.test_variance is not None:
                observed_pred = self.likelihood(
                    self.model(x_future_test), noise=self.test_variance
                )
            else:
                observed_pred = self.likelihood(self.model(x_future_test))

        if self.normalize_data:
            self.preds = (
                observed_pred.mean.numpy() * self.y_history_std
            ) + self.y_history_mean
            self.std = (
                observed_pred.variance.detach().numpy() ** 0.5 * self.y_history_std
            )
        else:
            self.preds = observed_pred.mean.numpy()
            self.std = observed_pred.variance.detach().numpy() ** 0.5

        scale = -scipy.stats.norm.ppf(self.alpha / 2)
        assert scale > 0
        self.preds_low = self.preds - self.std * scale
        self.preds_high = self.preds + self.std * scale

        return self.preds, self.std

    def predict_bound_models(self, x_history: np.ndarray, *args, **kwargs) -> None:
        """Predict values for both upper and lower confident interval's values.
        :param x_history: np.ndarray future points
        """
        warnings.warn("Not applicable for the GP model.")
