import numpy as np


class PCA:
    """Principal component analysis (PCA)

    Attributes
    ----------
    dim: int
        dimention size of input data
    eigen_value: ndarray (dim)
        eigen vaules of covariance matrix
    eigen_vector: ndarray (dim, dim)
        eigen vector of covariance matrix
    contribution_rate: ndarray (dim, )
        contribution rate of eigen value
    """

    def __init__(self, dim):
        self.dim = dim

        self.eigen_value = None
        self.eigen_vector = None
        self.contribution_rate = None

    def fit(self, X):
        """
        apply PCA to input data

        Parameters
        ----------
        X: ndarray (dim, n)
            input data

        Returns
        -------
        self: self instance
        """

        assert X.shape[1] == self.dim, f'The dimensions are different, expected {X.shape[1]}, but {self.dim} given'

        cov_matrics = np.cov(X.T, bias=True)
        self.eigen_value, self.eigen_vector = np.linalg.eig(cov_matrics)
        self.eigen_vector = self.eigen_vector

        contribution_rate = np.zeros(self.dim)
        for i in range(self.dim):
            contribution_rate[i] = self.eigen_value[i] / np.sum(self.eigen_value)
        self.contribution_rate = contribution_rate

        return self

    def transform(self, X):
        """
        transform input data with transformation matrics (eigen_vector)

        Parameters
        ----------
        X: ndarray (dim, n)
            input data

        Returns
        -------
        transformed: ndarray
            transformed data
        """

        transformed = np.dot(X, self.eigen_vector)
        return transformed

    def fit_transform(self, X):
        """
        apply PCA to input data and
        transform input data with transformation matrics (eigen_vector)

        Parameters
        ----------
        X: ndarray (dim, n)
            input data

        Returns
        -------
        transformed: ndarray
            transformed data
        """

        self = self.fit(X)
        transformed = np.dot(X, self.eigen_vector)
        return transformed
