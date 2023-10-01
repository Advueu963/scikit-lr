"""Base class for ensemble-based estimators."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from abc import ABC, abstractmethod
from copy import deepcopy
from numbers import Integral

# Third party
import numpy as np
from sklearn.ensemble._base import  BaseEstimator
from sklearn.base import MetaEstimatorMixin
# Local application
from ..utils.validation import check_random_state


# =============================================================================
# Constants
# =============================================================================

# Maximum random number that
# can be used for seeding
MAX_RAND_SEED = np.iinfo(np.int32).max


# =============================================================================
# Methods
# =============================================================================
def _indexes_to_mask(indexes, mask_length):
    """Convert list of indices to boolean mask."""
    # Initialize the mask of boolean values
    mask = np.zeros(mask_length, dtype=np.bool)

    # Set "True" in the given indexes
    mask[indexes] = True

    # Return the built mask
    return mask

def _predict_ensemble(estimator, X, sample_weight=None):
    """Predict using an ensemble-based estimator."""
    X = estimator._validate_data(X, reset=False)
    aggregate = estimator._rank_algorithm.aggregate

    ensemble_Y = [estimator.predict(X) for estimator in estimator.estimators_]

    # permutate ensemble_y axes according to axes
    # --> Groups ensemble predictions
    axes = (1, 0, 2)
    ensemble_Y = np.transpose(ensemble_Y, axes)

    Y = [aggregate(Y, sample_weight) for Y in ensemble_Y]

    return np.array(Y)