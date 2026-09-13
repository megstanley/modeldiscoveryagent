"""Small PyTorch SBI examples, not full flows. They are a TODO.

NPE (neural posterior estimation): a linear mean head and learned full covariance define q(theta | y, fixed x).
NRE (neural ratio estimation): a quadratic-feature logistic network learns pairwise prior-predictive ratios.
Both families suit the linear-Gaussian toy model; generic simulation does not make
these approximations universally accurate. Neither uses exact posterior targets.
For another ordered input history we retrain. Richer neural density models remain future work.
"""
from dataclasses import dataclass
from itertools import combinations
import numpy as np
import torch
from torch import nn

from ..beliefs import Beliefs, GaussianNeuralPosterior, Updater
from ..comparison import model_probabilities
from ..hypotheses import PolynomialHypothesis
from ..programs import as_designs, observation_hypothesis, hypothesis_fingerprint


class ConditionalGaussian(nn.Module):
    """Inspectable NPE: linear conditional mean and a shared learned Cholesky factor.

    A neural density estimator need not start as a deep network. For this toy,
    the exact conditional mean is linear and covariance does not depend on y.
    A richer family is needed for nonlinear means or multimodal posteriors.
    Outputs use standardized parameter coordinates, restored by condition_npe.
    """
    def __init__(self, n_observations, n_parameters):
        super().__init__()
        self.mean_head = nn.Linear(n_observations, n_parameters, dtype=torch.float64)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        self.raw_cholesky = nn.Parameter(torch.zeros(n_parameters, n_parameters, dtype=torch.float64))

    def scale_tril(self):
        """Exponentiate the diagonal to make covariance positive definite."""
        raw = self.raw_cholesky
        return torch.tril(raw, diagonal=-1) + torch.diag(torch.exp(torch.diag(raw)))

    def forward(self, standardized_y):
        """Return q(theta_standardized | y), not a distribution over observations."""
        return torch.distributions.MultivariateNormal(
            self.mean_head(standardized_y), scale_tril=self.scale_tril())


class RatioNetwork(nn.Module):
    """Balanced binary classifier with quadratic features; logit estimates log BF_ij.

    Class 1 is model i, class 0 is model j. Training parameters are prior draws,
    so the classifier compares marginal observation densities, not likelihoods
    evaluated at fitted parameters. Quadratic features match Gaussian log ratios.
    We use ordinary cross entropy for teaching, NOT the paper's exponential loss.
    """
    def __init__(self, dimension):
        super().__init__()
        rows, cols = torch.triu_indices(dimension, dimension)
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)
        self.head = nn.Linear(dimension + len(rows), 1, dtype=torch.float64)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, y):
        """Positive logits favour the numerator (class 1) model."""
        features = torch.cat([y, y[:, self.rows]*y[:, self.cols]], dim=1)
        return self.head(features).squeeze(-1)


@dataclass
class NeuralFit:
    """A reusable estimator for ONE ordered design list, with held-out diagnostics."""
    network: nn.Module
    input_mean: np.ndarray
    input_scale: np.ndarray
    train_losses: list[float]
    validation_loss: float
    simulations: int
    parameter_center: np.ndarray | None = None
    parameter_scale: np.ndarray | None = None
    hypothesis_key: str | None = None
    sampling_hypothesis: object | None = None
    designs: tuple = ()
    output_shape: tuple = ()

    def tensor(self, y):
        """Standardize using training data statistics, never the observed answer."""
        return torch.as_tensor((np.asarray(y)-self.input_mean)/self.input_scale,
                               dtype=torch.float64)


def _simulate_pairs(hypothesis, inputs, sigma, n, rng):
    
    if not isinstance(n, int) or isinstance(n, bool) or n < 20:
        raise ValueError("Use at least 20 simulated datasets.")
    designs = as_designs(inputs)
    generator = observation_hypothesis(hypothesis, sigma)
    theta = np.asarray(generator.sample_prior(rng, n), float)
    if theta.shape != (n, generator.n_params) or not np.all(np.isfinite(theta)):
        raise ValueError("Prior returned malformed parameter draws.")
    datasets = [np.asarray(generator.sample_observations(designs, t, rng), float) for t in theta]
    shape = datasets[0].shape
    if (len(shape) < 2 or shape[0] != len(designs) or any(s == 0 for s in shape)
            or any(y.shape != shape or not np.all(np.isfinite(y)) for y in datasets)):
        raise ValueError("Simulations must have a fixed, finite (designs, *output_shape) layout.")
    return theta, np.stack(datasets).reshape(n, -1), shape


def simulate_training_pairs(hypothesis, inputs, sigma, n, rng):
    """Draw labelled synthetic datasets; no observations are appended to real history.
    """
    theta, y, _ = _simulate_pairs(hypothesis, inputs, sigma, n, rng)
    return theta, y


def _optimize(network, loss, max_iter):
    """Full-batch L-BFGS; record closure evaluations, not mislabeled epochs."""
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    optimizer = torch.optim.LBFGS(network.parameters(), lr=0.8, max_iter=max_iter,
                                 line_search_fn="strong_wolfe", tolerance_grad=1e-7)
    losses = []

    def closure():
        optimizer.zero_grad()
        value = loss()
        if not torch.isfinite(value):
            raise ArithmeticError("Nonfinite neural training loss.")
        value.backward()
        losses.append(float(value.detach()))
        return value

    optimizer.step(closure)
    network.eval()
    return losses


def train_npe(hypothesis, inputs, sigma=None, *, n_simulations=4000, seed=0,
              max_iter=180) -> NeuralFit:
    """Fit a normalized Gaussian conditional density by negative log probability.

    Each label theta was used to produce its synthetic y. No exact fitting or
    evidence function is called. A disjoint 20% validation split is not optimized.
    Polynomial parameters use their known prior SD. Program parameters use the
    training split's mean and SD. Loss is in these standardized coordinates;
    comparing different parameterizations requires care. It is not model evidence.
    """
    designs = as_designs(inputs)
    theta, y, output_shape = _simulate_pairs(hypothesis, designs, sigma, n_simulations,
                                            np.random.default_rng(seed))
    split = int(0.8*n_simulations)
    mean, scale = y[:split].mean(0), np.maximum(y[:split].std(0), 1e-8)
    x = torch.as_tensor((y-mean)/scale, dtype=torch.float64)
    # Preserve the earlier polynomial coordinates; other priors need no prior_scale.
    if isinstance(hypothesis, PolynomialHypothesis):
        parameter_center = np.zeros(hypothesis.n_params)
        parameter_scale = np.full(hypothesis.n_params, hypothesis.prior_scale)
    else:
        parameter_center = theta[:split].mean(0)
        parameter_scale = np.maximum(theta[:split].std(0), 1e-8)
    target = torch.as_tensor((theta-parameter_center)/parameter_scale, dtype=torch.float64)
    # Initializer randomness is isolated; all trainable entries are set to zero.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        network = ConditionalGaussian(y.shape[1], hypothesis.n_params)
    losses = _optimize(network, lambda: -network(x[:split]).log_prob(target[:split]).mean(), max_iter)
    with torch.no_grad():
        validation = float(-network(x[split:]).log_prob(target[split:]).mean())
    return NeuralFit(network, mean, scale, losses, validation, n_simulations,
                     parameter_center, parameter_scale, hypothesis_fingerprint(hypothesis),
                     observation_hypothesis(hypothesis, sigma), designs, output_shape)


def condition_npe(fit, hypothesis, outputs) -> GaussianNeuralPosterior:
    """Evaluate at observations in fit.designs order, then restore parameter units.

    Shape checks cannot detect reordered values: the caller must preserve the
    recorded design order. A new design history requires retraining this estimator.
    """
    if fit.hypothesis_key != hypothesis_fingerprint(hypothesis):
        raise ValueError("NPE was trained for a different hypothesis or prior.")
    outputs = np.asarray(outputs, float)
    if outputs.shape not in (fit.input_mean.shape, fit.output_shape) or not np.all(np.isfinite(outputs)):
        raise ValueError("Outputs must match the estimator's ordered design list.")
    outputs = outputs.reshape(-1)
    with torch.no_grad():
        distribution = fit.network(fit.tensor(outputs[None, :]))
        mean = distribution.mean[0].numpy()*fit.parameter_scale+fit.parameter_center
        covariance = distribution.covariance_matrix[0].numpy()*np.outer(fit.parameter_scale, fit.parameter_scale)
    return GaussianNeuralPosterior(fit.sampling_hypothesis, mean, covariance)


def train_ratio(numerator, denominator, inputs, sigma=None, *, n_simulations=4000,
                seed=0, max_iter=180) -> NeuralFit:
    """Fit log p(y|numerator) / p(y|denominator) with balanced simulated classes.

    A small weight penalty avoids unbounded coefficients for nearly separable
    finite samples; this also biases ratios. Validation loss is not a guarantee
    of accurate Bayes factors on a particular observed dataset.
    """
    rng = np.random.default_rng(seed)
    designs = as_designs(inputs)
    _, yi, shape_i = _simulate_pairs(numerator, designs, sigma, n_simulations, rng)
    _, yj, shape_j = _simulate_pairs(denominator, designs, sigma, n_simulations, rng)
    if shape_i != shape_j:
        raise ValueError("Model comparison requires matching observation layouts.")
    split = int(0.8*n_simulations)
    y_train = np.concatenate([yi[:split], yj[:split]])
    y_valid = np.concatenate([yi[split:], yj[split:]])
    mean, scale = y_train.mean(0), np.maximum(y_train.std(0), 1e-8)
    x = torch.as_tensor((y_train-mean)/scale, dtype=torch.float64)
    valid = torch.as_tensor((y_valid-mean)/scale, dtype=torch.float64)
    labels = torch.cat([torch.ones(split), torch.zeros(split)]).double()
    valid_labels = torch.cat([torch.ones(n_simulations-split), torch.zeros(n_simulations-split)]).double()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        network = RatioNetwork(yi.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    losses = _optimize(network, lambda: criterion(network(x), labels)
                       + 1e-5*network.head.weight.square().mean(), max_iter)
    with torch.no_grad():
        validation = float(criterion(network(valid), valid_labels))
    return NeuralFit(network, mean, scale, losses, validation, 2*n_simulations,
                     designs=designs, output_shape=shape_i)


def ratio_log_bayes_factor(fit, outputs):
    """Read the logit with its documented numerator/denominator sign."""
    outputs = np.asarray(outputs, float)
    if outputs.shape not in (fit.input_mean.shape, fit.output_shape) or not np.all(np.isfinite(outputs)):
        raise ValueError("Outputs must match the ratio estimator's ordered designs.")
    with torch.no_grad():
        return float(fit.network(fit.tensor(outputs.reshape(1, -1)))[0])


def reconcile_log_ratios(names, ratios):
    """Least-squares relative log evidences with sum zero, plus cycle residuals.

    ratios maps (numerator_name, denominator_name) to log BF. Require all pairs
    exactly once. An additive evidence constant is unidentifiable: these outputs
    must NOT be presented as absolute log evidences.
    """
    if not names or len(set(names)) != len(names):
        raise ValueError("Supply unique model names.")
    expected = {frozenset(pair) for pair in combinations(names, 2)}
    if ({frozenset(pair) for pair in ratios} != expected or len(ratios) != len(expected)
            or any(not np.isfinite(value) for value in ratios.values())):
        raise ValueError("Supply one finite log ratio per unordered model pair.")
    matrix, values = [], []
    for (numerator, denominator), value in ratios.items():
        row = np.zeros(len(names))
        row[names.index(numerator)], row[names.index(denominator)] = 1, -1
        matrix.append(row)
        values.append(value)
    if not matrix:
        return {names[0]: 0.0}, np.array([])
    matrix, values = np.asarray(matrix), np.asarray(values)
    relative = np.linalg.lstsq(np.vstack([matrix, np.ones(len(names))]),
                               np.append(values, 0.0), rcond=None)[0]
    return dict(zip(names, relative)), values-matrix @ relative


class NeuralUpdater(Updater):
    """NPE plus separately configured model comparison; refit for each input history.

    evidence='ratios' uses learned pairwise ratios; evidence='exact' is explicitly
    an ablation isolating NPE error, not a likelihood-free method. Models and
    measurement noise are supplied, never learned from hidden world answers.
    This deliberately retrains rather than reusing a network on mismatched inputs.
    """
    def __init__(self, sigma=None, *, evidence="ratios", n_simulations=4000, seed=0, max_iter=180):
        if evidence not in {"ratios", "exact"}:
            raise ValueError("evidence must be ratios or exact.")
        self.sigma, self.evidence = sigma, evidence
        self.n_simulations, self.max_iter = n_simulations, max_iter
        self.rng = np.random.default_rng(seed)
        self.last_npe, self.last_ratios = {}, {}
        self.last_relative_log_evidences = {}
        self.last_ratio_residuals = np.array([])

    def update(self, data, hypotheses) -> Beliefs:
        """Train on simulations at current settings, then condition on measured y."""
        names = [h.name for h in hypotheses]
        if not names or len(set(names)) != len(names):
            raise ValueError("Supply a nonempty pool with unique names.")
        if not data:
            raise ValueError("Neural inference requires observed data at a fixed design layout.")
        if self.evidence == "exact" and any(not isinstance(h, PolynomialHypothesis) for h in hypotheses):
            raise ValueError("Exact evidence is only implemented for polynomial hypotheses.")
        inputs = tuple(obs.design for obs in data)
        outputs = np.stack([np.asarray(obs.y, float) for obs in data])
        options = dict(n_simulations=self.n_simulations, max_iter=self.max_iter)
        def seed():
            return int(self.rng.integers(0, 2**31-1))
        self.last_npe = {h.name: train_npe(h, inputs, self.sigma, seed=seed(), **options)
                         for h in hypotheses}
        posteriors = {h.name: condition_npe(self.last_npe[h.name], h, outputs)
                      for h in hypotheses}
        self.last_ratios = {}
        if self.evidence == "exact":
            from .exact_gaussian import log_model_evidence
            log_z = {h.name: log_model_evidence(h, data, self.sigma if self.sigma is not None else h.noise.sigma)
                     for h in hypotheses}
            weights = model_probabilities(log_z)
            self.last_relative_log_evidences = {}
            self.last_ratio_residuals = np.array([])
        else:
            ratios = {}
            for hi, hj in combinations(hypotheses, 2):
                pair = (hi.name, hj.name)
                fit = train_ratio(hi, hj, inputs, self.sigma, seed=seed(), **options)
                self.last_ratios[pair] = fit
                ratios[pair] = ratio_log_bayes_factor(fit, outputs)
            relative, residuals = reconcile_log_ratios(names, ratios)
            self.last_relative_log_evidences = relative
            self.last_ratio_residuals = residuals
            weights, log_z = model_probabilities(relative), None
        return Beliefs("npe+"+self.evidence, "Learned toy approximation within this pool; inspect validation and reference errors.",
                       posteriors, weights, log_z, "bayesian_model_probability")
