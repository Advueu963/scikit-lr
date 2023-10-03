"""Weight Boosting.

This module contains weight boosting estimators for both Label Ranking.

The module structure is the following:

- The ``BaseWeightBoosting`` base class implements a common ``fit`` method
  for all the estimators in the module. Label Ranking and Partial Label
  Ranking only differ from each other in the loss function that is optimized.

- ``AdaBoostLabelRanker`` implements adaptive boosting for
  Label Ranking problems.
"""


# =============================================================================
# Imports
# =============================================================================

# Standard
from abc import ABC, abstractmethod
from numbers import Integral, Real

# Third party
import numpy as np
from sklearn.ensemble._weight_boosting import BaseWeightBoosting
from sklearn.utils.validation import _num_samples
from sklearn.utils import _safe_indexing
from sklr.metrics import kendall_distance

# Local application
from ..base import LabelRankerMixin
from ..tree import DecisionTreeLabelRanker
from ..utils.validation import has_fit_parameter

# =============================================================================
# AdaBoost Label Ranker
# =============================================================================
class AdaBoostLabelRanker(LabelRankerMixin, BaseWeightBoosting):
    """An AdaBoost Label Ranker.

    AdaBoost [1] Label Ranker is a meta-estimator that begins by fitting
    a Label Ranker on the original dataset and then fits additional copies
    of the Label Ranker on the same dataset but where the weights of
    incorrectly classified instances are adjusted such that subsequent
    Label Rankers focus more on difficult cases.

    Hyperparameters
    ---------------
    base_estimator : object, optional (default=None)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required. If ``None``, then
        the base estimator is ``DecisionTreeLabelRanker(max_depth=3)``.

    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.

    learning_rate : float, optional (default=1.0)
        Learning rate shrinks the contribution of each estimator by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.

    random_state : {int, RandomState instance, None}, optional (default=None)
        - If int, random_state is the seed used by the random number generator.
        - If RandomState instance, random_state is the random number generator.
        - If None, the random number generator is the RandomState instance used
          by `np.random`.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted sub-estimators.

    n_samples_ : int
        The number of samples when ``fit`` is performed.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_classes_ : int
        The number of classes when ``fit`` is performed.

    estimator_weights_ : np.ndarray of shape (n_estimators)
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : np.ndarray of shape (n_estimators)
        Error for each estimator in the boosted ensemble.

    See also
    --------
    DecisionTreeLabelRanker, AdaBoostPartialLabelRanker

    References
    ----------
    .. [1] `Y. Freund and R. Schapire, "A Decision-Theoretic Generalization of
            On-Line Learning and an Application to Boosting", Journal of
            Computer and System Sciences, vol. 55, pp.119-139, 1997.`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.ensemble import AdaBoostLabelRanker
    >>> X = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0]])
    >>> Y = np.array([[1, 2, 3], [2, 1, 3], [1, 2, 3], [3, 1, 2]])
    >>> model = AdaBoostLabelRanker(random_state=0)
    >>> clf = model.fit(X, Y)
    >>> clf.predict(np.array([[0, 1, 0]]))
    np.array([[2, 1, 3]])
    """

    def __init__(self,
                 estimator=None,
                 n_estimators=50,
                 learning_rate=1.0,
                 random_state=None):
        """Constructor."""
        # Call to the constructor of the parent
        super(AdaBoostLabelRanker, self).__init__(estimator=estimator,
                         n_estimators=n_estimators,
                         learning_rate=learning_rate,
                         random_state=random_state)
    def fit(self, X, Y, sample_weight=None):
        X, self._Y = self._validate_data(X, Y, multi_output=True)
        # avoiding  the checks of sklearn
        y = Y[:,0]
        return super(AdaBoostLabelRanker, self).fit(X, y, sample_weight)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        # Call to the method of the parent using
        # as default estimator a Decision Tree
        # Label Ranker with a maximum depth of one
        super()._validate_estimator(
            default=DecisionTreeLabelRanker(max_depth=1))

        # Check that the estimator support sample
        # weighting, raising the corresponding
        # exception when it is not supported
        if not has_fit_parameter(self.estimator_, "sample_weight"):
            raise ValueError("{} does not support sample_weight."
                             .format(self.estimator_.__class__.__name__))

    def _boost(self, iboost, X, Y, sample_weight, random_state):
        estimator = self._make_estimator(random_state=random_state)

        # Weighted sampling of the training set with replacement
        bootstrap_idx = random_state.choice(
            np.arange(_num_samples(X)),
            size=_num_samples(X),
            replace=True,
            p=sample_weight,
        )

        # Fit on the bootstrapped sample and obtain a prediction
        # for all samples in the training set
        _X = _safe_indexing(X, bootstrap_idx)
        _Y = _safe_indexing(self._Y, bootstrap_idx)
        estimator.fit(_X, _Y)
        Y_predict = estimator.predict(X)

        error_vect = np.array(
            [kendall_distance(self._Y[sample, None], Y_predict[sample, None], normalize=True) for sample in
             range(X.shape[0])])
        sample_mask = sample_weight > 0
        masked_sample_weight = sample_weight[sample_mask]
        masked_error_vector = error_vect[sample_mask]

        error_max = np.max(masked_error_vector)
        if error_max != 0:
            masked_error_vector /= error_max

        # Calculate the average loss
        estimator_error = np.sum(masked_sample_weight * masked_error_vector)

        if estimator_error <= 0:
            # Stop if fit is perfect
            return sample_weight, 1.0, 0.0

        elif estimator_error >= 0.5:
            # Discard current estimator only if it isn't the only one
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)
            return None, None, None

        beta = estimator_error / (1.0 - estimator_error)

        # Boost weight using AdaBoost.R2 alg
        estimator_weight = self.learning_rate * np.log(1.0 / beta)

        if not iboost == self.n_estimators - 1:
            sample_weight[sample_mask] *= np.power(
                beta, (1.0 - masked_error_vector) * self.learning_rate
            )

        return sample_weight, estimator_weight, estimator_error

    def predict(self, X):
        limit = len(self.estimators_)
        aggregate = self._rank_algorithm.aggregate
        n_samples = X.shape[0]
        Y = np.array([estimator.predict(X) for estimator in self.estimators_])
        Y = [aggregate(Y[:, sample], self.estimator_weights_[:limit]) for sample in range(n_samples)]

        return np.array(Y)