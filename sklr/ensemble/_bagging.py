"""Bagging meta-estimator."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from abc import ABC, abstractmethod
from numbers import Real, Integral

# Third party
import numpy as np
from sklearn.utils.validation import  check_is_fitted
from sklearn.ensemble._bagging import  BaseBagging

# Local application
from ._base import _indexes_to_mask
from ._base import MAX_RAND_SEED
from ..base import LabelRankerMixin, PartialLabelRankerMixin
from ..tree import DecisionTreeLabelRanker, DecisionTreePartialLabelRanker
from ._base import _predict_ensemble
# =============================================================================
# Bagging Label Ranker
# =============================================================================
class BaggingLabelRanker(LabelRankerMixin, BaseBagging):
    """A Bagging Label Ranker.

    A Bagging Label Ranker is an ensemble meta-estimator that fits base
    Label Rankers each on random subsets of the original dataset and then
    aggregate their individual predictions to form a final prediction.
    Such a meta-estimator can typically be used as a way to reduce the
    variance of a black-box estimator (e.g., a decision tree), by introducing
    randomization into its construction procedure and then making an
    ensemble out of it.

    This algorithm encompasses several works from the literature. When random
    subsets of the dataset are drawn as random subsets of the samples, then
    this algorithm is known as Pasting [1]_. If samples are drawn with
    replacement, then the method is known as Bagging [2]_. When random subsets
    of the dataset are drawn as random subsets of the features, then the method
    is known as Random Subspaces [3]_. Finally, when base estimators are built
    on subsets of both samples and features, then the method is known as
    Random Patches [4]_.

    Hyperparamters
    --------------
    base_estimator : {object, None}, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : {int, float}, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.

            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : {int, float}, optional (default=1.0)
        The number of features to draw from X to train each base estimator.

        - If int, then draw `max_features` features.
        - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : bool, optional (default=True)
        Whether samples are drawn with replacement. If False, sampling
        without replacement is performed.

    bootstrap_features : bool, optional (default=False)
        Whether features are drawn with replacement.

    random_state : {int, RandomState instance, None}, optional (default=None)
        - If int, random_state is the seed used by the random number generator.
        - If RandomState instance, random_state is the random number generator.
        - If None, the random number generator is the RandomState instance used
          by `np.random`.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    n_samples_ : int
        The number of samples when ``fit`` is performed.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_classes_ : int
        The number of classes when ``fit`` is performed.

    estimators_ : list of estimators
        The collection of fitted base estimators.

    estimators_features_ : list of np.ndarrays
        The subset of drawn features for each base estimator.

    See also
    --------
    BaggingPartialLabelRanker

    References
    ----------
    .. [1] `L. Breiman, "Pasting small votes for classification in large
            databases and on-line", Machine Learning, vol. 36, pp. 85-103,
            1999.`_

    .. [2] `L. Breiman, "Bagging predictors", Machine Learning, vol. 24,
            pp. 123-140, 1996.`_

    .. [3] `T. Ho, "The random subspace method for constructing decision
            forests", Pattern Analysis and Machine Intelligence, vol. 20,
            pp. 832-844, 1998.`_

    .. [4] `G. Louppe and P. Geurts, "Ensembles on Random Patches", In
            Proceedings of teh Joint European Conference on Machine Learning
            and Knowledge Discovery in Databases, 2012, pp. 346-361.`_

    .. [5] `Juan A. Aledo, José A. Gámez and D. Molina, "Tackling the
            supervised label ranking problem by bagging weak learners",
            Information Fusion, vol. 35, pp. 38-50, 2017.`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.ensemble import BaggingLabelRanker
    >>> X = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0]])
    >>> Y = np.array([[1, 2, 3], [2, 1, 3], [1, 2, 3], [3, 1, 2]])
    >>> model = BaggingLabelRanker(random_state=0)
    >>> clf = model.fit(X, Y)
    >>> clf.predict(np.array([[0, 1, 0]]))
    np.array([[2, 1, 3]])
    """


    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 random_state=None):
        """Constructor."""
        # Call to the constructor of the parent
        super(BaggingLabelRanker,self).__init__(base_estimator,
                         n_estimators=n_estimators,
                         max_samples=max_samples,
                         max_features=max_features,
                         bootstrap=bootstrap,
                         bootstrap_features=bootstrap_features,
                         random_state=random_state)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        # Call to the method of the parent with a Decision
        # Tree Label Ranking as default estimator
        super(BaggingLabelRanker,self)._validate_estimator(default=DecisionTreeLabelRanker())

    def _set_oob_score(self, X, y):
        pass

    def predict(self, X):
        return _predict_ensemble(self,X)

# =============================================================================
# Bagging Partial Label Ranker
# =============================================================================
class BaggingPartialLabelRanker(PartialLabelRankerMixin, BaseBagging):
    """A Bagging Partial Label Ranker.

    A Bagging Partial Label Ranker is an ensemble meta-estimator that fits base
    Partial Label Rankers each on random subsets of the original dataset and
    then aggregate their individual predictions to form a final prediction.
    Such a meta-estimator can typically be used as a way to reduce the variance
    of a black-box estimator (e.g., a decision tree), by introducing
    randomization into its construction procedure and then making an ensemble
    out of it.

    This algorithm encompasses several works from the literature. When random
    subsets of the dataset are drawn as random subsets of the samples, then
    this algorithm is known as Pasting [1]_. If samples are drawn with
    replacement, then the method is known as Bagging [2]_. When random subsets
    of the dataset are drawn as random subsets of the features, then the method
    is known as Random Subspaces [3]_. Finally, when base estimators are built
    on subsets of both samples and features, then the method is known as
    Random Patches [4]_.

    Hyperparamters
    --------------
    base_estimator : {object, None}, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : {int, float}, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.

            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : {int, float}, optional (default=1.0)
        The number of features to draw from X to train each base estimator.

        - If int, then draw `max_features` features.
        - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : bool, optional (default=True)
        Whether samples are drawn with replacement. If False, sampling
        without replacement is performed.

    bootstrap_features : bool, optional (default=False)
        Whether features are drawn with replacement.

    random_state : {int, RandomState instance, None}, optional (default=None)
        - If int, random_state is the seed used by the random number generator.
        - If RandomState instance, random_state is the random number generator.
        - If None, the random number generator is the RandomState instance used
          by `np.random`.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    n_samples_ : int
        The number of samples when ``fit`` is performed.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_classes_ : int
        The number of classes when ``fit`` is performed.

    estimators_ : list of estimators
        The collection of fitted base estimators.

    estimators_features_ : list of np.ndarrays
        The subset of drawn features for each base estimator.

    See also
    --------
    BaggingLabelRanker

    References
    ----------
    .. [1] `L. Breiman, "Pasting small votes for classification in large
            databases and on-line", Machine Learning, vol. 36, pp. 85-103,
            1999.`_

    .. [2] `L. Breiman, "Bagging predictors", Machine Learning, vol. 24,
            pp. 123-140, 1996.`_

    .. [3] `T. Ho, "The random subspace method for constructing decision
            forests", Pattern Analysis and Machine Intelligence, vol. 20,
            pp. 832-844, 1998.`_

    .. [4] `G. Louppe and P. Geurts, "Ensembles on Random Patches", In
            Proceedings of teh Joint European Conference on Machine Learning
            and Knowledge Discovery in Databases, 2012, pp. 346-361.`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.ensemble import BaggingPartialLabelRanker
    >>> X = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0]])
    >>> Y = np.array([[1, 2, 3], [2, 1, 3], [1, 2, 3], [3, 1, 2]])
    >>> model = BaggingPartialLabelRanker(random_state=0)
    >>> clf = model.fit(X, Y)
    >>> clf.predict(np.array([[0, 1, 0]]))
    np.array([[1, 1, 2]])
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 random_state=None):
        """Constructor."""
        # Call to the constructor of the parent
        super().__init__(base_estimator,
                         n_estimators=n_estimators,
                         max_samples=max_samples,
                         max_features=max_features,
                         bootstrap=bootstrap,
                         bootstrap_features=bootstrap_features,
                         random_state=random_state)

    def _set_oob_score(self, X, y):
        pass

    def predict(self, X):
        return _predict_ensemble(self,X)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        # Call to the method of the parent with a Decision
        # Tree Partial Label Ranking as base estimator
        super(BaggingPartialLabelRanker,self)._validate_estimator(default=DecisionTreePartialLabelRanker())
