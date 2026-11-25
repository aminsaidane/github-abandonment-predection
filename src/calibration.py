"""Small probability calibrators with a fit / predict interface (picklable, light to import).

Kept apart from final_evaluate.py so that the dashboard can load the saved model without pulling in
matplotlib and the evaluation code.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


class Identity:
    """No recalibration."""
    name = "none"

    def fit(self, p, y):
        return self

    def predict(self, p):
        return np.asarray(p, dtype=float)


class Platt:
    """Sigmoid recalibration: logistic regression on the logit of the model's probability."""
    name = "sigmoid"

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
        return np.log(p / (1 - p))[:, None]

    def fit(self, p, y):
        self.model_ = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(p), y)
        return self

    def predict(self, p):
        return self.model_.predict_proba(self._logit(p))[:, 1]


class Isotonic:
    """Monotone step-function recalibration."""
    name = "isotonic"

    def fit(self, p, y):
        self.model_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
        return self

    def predict(self, p):
        return self.model_.predict(np.asarray(p, dtype=float))


CALIBRATORS = {"none": Identity, "sigmoid": Platt, "isotonic": Isotonic}


def make_calibrator(name):
    return CALIBRATORS[name]()
