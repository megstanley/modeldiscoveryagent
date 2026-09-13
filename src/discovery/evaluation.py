"""Checks on model adequacy and forecasts; this is never used in model choices.

The caller owns held-out answers. Keep those out of Designer and Updater inputs.
Scores on training residuals diagnose fitted adequacy, not independent test error.
"""
import numpy as np

from .beliefs import ParameterPosterior
from .records import Dataset, observed_arrays


def mean_squared_error(prediction, target) -> float:
    """Mean squared error for equally shaped finite non-empty arrays."""
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    if (prediction.shape != target.shape or not prediction.size
            or not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(target))):
        raise ValueError("Forecasts and targets must be finite, non-empty, and equally shaped.")
    return float(np.mean((prediction - target)**2))


def residuals_in_noise_sds(fit: ParameterPosterior, data: Dataset, sigma: float) -> np.ndarray:
    """Return (observed - posterior-mean prediction) / measurement SD for each record.

    On the training history these are fitted residual diagnostics, not calibrated
    posterior-predictive z scores: parameter uncertainty is not in the denominator.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Measurement SD must be positive and finite.")
    inputs, outputs = observed_arrays(data)
    predictions, _ = fit.predict(inputs)
    return (outputs - predictions) / sigma
