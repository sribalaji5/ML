import numpy as np

class GMM_EM:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n_samples, n_features = X.shape

        # Initialize parameters randomly
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = np.random.randn(self.n_components, n_features)
        self.covariances = np.array([np.eye(n_features)] * self.n_components)

        for _ in range(self.max_iter):
            # E-step: calculate responsibilities
            resp = self._expectation(X)

            # M-step: update parameters
            self._maximization(X, resp)

            # Check for convergence
            if np.abs(np.log(self.prev_likelihood) - np.log(self._likelihood(X))) < self.tol:
                break

    def _expectation(self, X):
        n_samples = X.shape[0]
        resp = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            mean = self.means[k]
            cov = self.covariances[k]
            weight = self.weights[k]

            resp[:, k] = weight * self._gaussian_pdf(X, mean, cov)

        # Normalize responsibilities
        resp /= resp.sum(axis=1, keepdims=True)
        return resp

    def _maximization(self, X, resp):
        n_samples = X.shape[0]

        # Update weights
        self.weights = resp.mean(axis=0)

        # Update means
        weighted_sum = np.dot(resp.T, X)
        self.means = weighted_sum / resp.sum(axis=0, keepdims=True)

        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means[k]
            cov = np.dot(resp[:, k] * diff.T, diff) / resp[:, k].sum()
            self.covariances[k] = cov

        # Calculate log-likelihood for convergence check
        self.prev_likelihood = self._likelihood(X)

    def _likelihood(self, X):
        likelihood = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            likelihood[:, k] = self.weights[k] * self._gaussian_pdf(X, self.means[k], self.covariances[k])
        return np.log(likelihood.sum(axis=1)).sum()

    def _gaussian_pdf(self, X, mean, cov):
        n_features = X.shape[1]
        det_cov = np.linalg.det(cov)
        norm_const = 1.0 / np.sqrt((2 * np.pi) ** n_features * det_cov)
        inv_cov = np.linalg.inv(cov)
        exponent = -0.5 * np.sum(np.dot(X - mean, inv_cov) * (X - mean), axis=1)
        return norm_const * np.exp(exponent)

# Example usage:
if __name__ == "__main__":
    # Generate some random data
    np.random.seed(0)
    n_samples = 1000
    mean1 = [0, 0]
    cov1 = [[1, 0.5], [0.5, 1]]
    mean2 = [5, 5]
    cov2 = [[1, -0.5], [-0.5, 1]]
    X1 = np.random.multivariate_normal(mean1, cov1, n_samples)
    X2 = np.random.multivariate_normal(mean2, cov2, n_samples)
    X = np.vstack([X1, X2])

    # Fit GMM using EM algorithm
    gmm = GMM_EM(n_components=2)
    gmm.fit(X)

    # Print parameters
    print("Weights:", gmm.weights)
    print("Means:", gmm.means)
    print("Covariances:", gmm.covariances)
