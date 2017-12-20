import math
import numpy as np
from scipy.special import expit


class LogReg():
    def __init__(self, lambda_1=0.0, lambda_2=1.0, gd_type='full',
                 tolerance=1e-4, max_iter=1000, w0=None, alpha=1e-3):
        """
        lambda_1: L1 regularization param
        lambda_2: L2 regularization param
        gd_type: 'full' or 'stochastic'
        tolerance: for stopping gradient descent
        max_iter: maximum number of steps in gradient descent
        w0: np.array of shape (d) - init weights
        alpha: learning rate
        """
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.gd_type = gd_type
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.w0 = w0
        self.alpha = alpha
        self.w = None
        self.loss_history = None

    def fit(self, X, y):
        """
        X: np.array of shape (l, d)
        y: np.array of shape (l)
        ---
        output: self
        """
        self.loss_history = []
        count = 0
        self.w = np.ones((len(X[0]),))
        while np.linalg.norm(self.w - self.w0) > self.tolerance or count < self.max_iter:
            count += 1
            self.w = self.w0
            if self.gd_type == 'stochastic':
                i = random.randint(0, len(y))
                grad = self.calc_gradient(X[i, :], np.array(y[i]))
            else:
                grad = self.calc_gradient(X, y)
            self.w = self.w0 - self.alpha * grad
            loss = self.calc_loss(X, y)
            self.loss_history.append(loss)
            self.w0 = self.w
        return self

    def predict_proba(self, X):
        """
        X: np.array of shape (l, d)
        ---
        output: np.array of shape (l, 2) where
        first column has probabilities of -1
        second column has probabilities of +1
        """
        if self.w is None:
            raise Exception('Not trained yet')
        pred = expit(np.dot(X, self.w))
        return pred

    def calc_gradient(self, X, y):
        """
        X: np.array of shape (l, d) (l can be equal to 1 if stochastic)
        y: np.array of shape (l)
        ---
        output: np.array of shape (d)
        """
        tm1 = expit(-y * np.dot(X, self.w))
        tm2 = y[:, np.newaxis] * X
        tm3 = tm1[:, np.newaxis] * tm2
        tm4 = -np.sum(tm3, axis=0)
        grad = tm4 / X.shape[0] + self.lambda_2 * self.w
        return grad

    def calc_loss(self, X, y):
        """
        X: np.array of shape (l, d)
        y: np.array of shape (l)
        ---
        output: float
        """
        n = X.shape[0]
        tm1 = np.logaddexp(0, -y * np.dot(X, self.w))
        reg = self.lambda_2 * np.sum(self.w ** 2) / 2
        loss = (1 / n) * np.sum(tm1, axis=0) + reg
        return loss
