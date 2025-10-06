import numpy as np
from scipy import spatial


class MegKDE:
    """Matched Elliptical Gaussian Kernel Density Estimator

    Adapted from the algorithm specified in the BAMBIS's model specified Wolf 2017
    to support weighted samples.
    """

    def __init__(
        self,
        train: np.ndarray,
        weights: np.ndarray | None = None,
        truncation: float = 3.0,
        nmin: int = 4,
        factor: float = 1.0,
    ):
        """
        Args:
            train (np.ndarray): The training data set. Should be a 1D array of samples or a 2D array
                of shape (n_samples, n_dim).
            weights (np.ndarray, optional): An array of weights. If not specified, equal weights are assumed.
            truncation (float, optional): The maximum deviation (in sigma) to use points in the KDE
            nmin (int, optional): The minimum number of points required to estimate the density
            factor (float, optional): Send bandwidth to this factor of the data estimate
        """

        self.truncation = truncation
        self.nmin = nmin
        self.train = train
        if len(train.shape) == 1:
            train = np.atleast_2d(train).T
        self.num_points, self.num_dim = train.shape
        if weights is None:
            weights = np.ones(self.num_points)
        self.weights = weights

        self.mean = np.average(train, weights=weights, axis=0)
        dx = train - self.mean
        cov = np.atleast_2d(np.cov(dx.T, aweights=weights))
        self.A = np.linalg.cholesky(np.linalg.inv(cov))  # The sphere-ifying transform

        self.d = np.dot(dx, self.A)  # Sphere-ified data
        self.tree = spatial.cKDTree(self.d)  # kD tree of data

        self.sigma = 2.0 * factor * np.power(self.num_points, -1.0 / (4 + self.num_dim))  # Starting sigma (bw) of Gauss
        self.sigma_fact = -0.5 / (self.sigma * self.sigma)

        # Can't get normed probs to work atm, turning off for now as I don't need normed pdfs for contours
        # self.norm = np.product(np.diagonal(self.A)) * (2 * np.pi) ** (-0.5 * self.num_dim)  # prob norm
        # self.scaling = np.power(self.norm * self.sigma, -self.num_dim)

    def evaluate(self, data: np.ndarray) -> np.ndarray:
        """Estimate un-normalised probability density at target points

        Args:
            data (np.ndarray): 2D array of shape (n_samples, n_dim).

        Returns:
            np.ndarray: A `(n_samples)` length array of estimates

        """
        if len(data.shape) == 1 and self.num_dim == 1:
            data = np.atleast_2d(data).T

        _d = np.dot(data - self.mean, self.A)

        # Get all points within range of kernels
        neighbors = self.tree.query_ball_point(_d, self.sigma * self.truncation)  # type: ignore
        out = []
        for i, n in enumerate(neighbors):
            if len(n) >= self.nmin:
                diff = self.d[n, :] - _d[i]
                distsq = np.sum(diff * diff, axis=1)
            else:
                # If too few points get nmin closest
                dist, n = self.tree.query(_d[i], k=self.nmin)  # noqa: PLW2901 # TODO: I assume this isn't an error
                distsq = dist * dist
            out.append(np.sum(self.weights[n] * np.exp(self.sigma_fact * distsq)))
        return np.array(out)  # * self.scaling
