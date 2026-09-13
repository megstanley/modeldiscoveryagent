"""Parameter inference by prior importance sampling and tempered SMC.

Read in this order: ParticleTarget, importance_fit, tempered_smc, then the
model adapters. Polynomial calculations are isolated in polynomial_target;
the sampling algorithms never inspect a model's formula or prior family, 
they just access it through the log_prior and log_likelihood callables as required.

This is the within-model parameter/evidence component of a discovery agent.
It fits one fixed hypothesis to the full measured history, from its prior.

A separate caller does model proposal, experiment selection and pool management.
It is not the complete nested SMC algorithm in Murphy's Appendix A.3/A.6:
https://arxiv.org/html/2608.09696v4#A6

Current scope: fixed-dimensional continuous parameters and deterministic,
evaluable likelihoods, including every normalization constant. A declared
approximate likelihood targets that approximation, not the original model.
Simulation alone is insufficient. Stochastic particle-filter likelihood
estimates need a separate extended-state/pseudo-marginal treatment.
"""
from dataclasses import dataclass
from functools import partial
from typing import Callable, Sequence
import numpy as np
from scipy.special import logsumexp

from ..beliefs import Beliefs, ParticlePosterior, Updater
from ..comparison import model_probabilities
from ..hypotheses import Hypothesis, PolynomialHypothesis, design_matrix
from ..programs import observation_hypothesis
from ..records import Dataset, observed_arrays


@dataclass(frozen=True)
class ParticleTarget:
    """The model-specific inputs to both numerical algorithms.

    hypothesis supplies n_params and sample_prior(rng, n). Both log functions
    accept an (n, n_params) array and return n log densities. log_likelihood
    closes over the fixed dataset; log_prior describes the SAME normalized
    prior as sample_prior. Return -inf outside support, never NaN or +inf.

    At temperature beta the unnormalized target is:
        prior(theta) * likelihood(data | theta) ** beta.
    The evidence at beta=1 is the integral of prior times likelihood.
    """
    hypothesis: Hypothesis
    log_prior: Callable[[np.ndarray], np.ndarray]
    log_likelihood: Callable[[np.ndarray], np.ndarray]


@dataclass
class TemperingStep:
    """One numerical step on FIXED data, with pre-resampling weights exposed."""
    beta: float
    particles_before: np.ndarray
    log_likelihoods: np.ndarray
    weights: np.ndarray
    ess: float
    log_evidence_increment: float
    ancestors: np.ndarray
    particles_after: np.ndarray
    acceptance_rate: float


@dataclass
class ParticleFit:
    """Parameter posterior, log evidence and actual per-particle likelihood evaluations."""
    posterior: ParticlePosterior
    log_evidence: float
    steps: list[TemperingStep]
    likelihood_evaluations: int


# General inference: no polynomial or Gaussian-prior assumptions.

def importance_fit(target: ParticleTarget, *, n_particles, rng) -> ParticleFit:
    """Sample the prior, then weight by likelihood on the complete fixed dataset.

    Because the proposal IS the prior, the prior/proposal ratio cancels:
        w_i proportional to L(theta_i), Z_hat = mean_i L(theta_i).
    This is prior importance sampling, not IS from an arbitrary proposal.
    """
    theta, ell = _initial_particles(target, n_particles, rng)
    weights, log_z = _likelihood_weights(ell)
    posterior = ParticlePosterior(target.hypothesis, theta, weights)
    step = TemperingStep(1.0, theta.copy(), ell.copy(), weights.copy(),
                         effective_sample_size(weights), log_z,
                         np.arange(n_particles), theta.copy(), 0.0)
    return ParticleFit(posterior, log_z, [step], n_particles)


def tempered_smc(target: ParticleTarget, *, n_particles, rng,
                 temperatures=None, ess_fraction=0.8, moves=4,
                 proposal_scale=1.0, max_steps=200) -> ParticleFit:
    """Bridge from the prior (beta=0) to the posterior (beta=1).

    Each step has three parts:
      1. Weight by L(theta) ** (next_beta - beta), recording the evidence ratio.
      2. Resample these weights to obtain an equally weighted population.
      3. Move particles with Metropolis targeting prior * L ** next_beta.

    A supplied schedule must increase from 0 to 1. Otherwise choose temperatures
    by weight ESS. Proposal covariance is estimated from the weighted cloud,
    then held fixed during that step's moves. This population adaptation is a
    finite-particle approximation, not an exact independent posterior sampler.
    Check independent runs and references; high ESS alone does not prove accuracy.
    """
    _count(moves, "moves")
    _count(max_steps, "max_steps", minimum=1)
    if not 0 < ess_fraction < 1 or not np.isfinite(proposal_scale) or proposal_scale <= 0:
        raise ValueError("Use 0 < ess_fraction < 1 and positive finite proposal_scale.")
    schedule = _temperature_schedule(temperatures)
    theta, ell = _initial_particles(target, n_particles, rng)
    # Empirical prior scales regularize the proposal, without assuming a prior family.
    prior_variance = np.maximum(np.var(theta, axis=0), 1e-12)
    evaluations = n_particles
    beta, log_z, steps = 0.0, 0.0, []

    while beta < 1:
        if len(steps) >= max_steps:
            raise RuntimeError("SMC did not reach beta=1; increase max_steps or inspect likelihoods.")
        next_beta = (float(schedule[len(steps)+1]) if schedule is not None
                     else _next_temperature(beta, ell, ess_fraction))

        weights, log_ratio = _likelihood_weights((next_beta-beta)*ell)
        before, before_ell = theta.copy(), ell.copy()
        chol = _proposal_cholesky(theta, weights, prior_variance, proposal_scale) if moves else None
        ancestors = rng.choice(n_particles, n_particles, p=weights)
        theta, ell = theta[ancestors].copy(), ell[ancestors].copy()
        theta, ell, acceptance, work = _metropolis_moves(
            target, theta, ell, next_beta, chol, moves, rng)

        log_z += log_ratio
        evaluations += work
        steps.append(TemperingStep(next_beta, before, before_ell, weights.copy(),
                                    effective_sample_size(weights), log_ratio,
                                    ancestors, theta.copy(), acceptance))
        beta = next_beta

    posterior = ParticlePosterior(target.hypothesis, theta, np.full(n_particles, 1/n_particles))
    return ParticleFit(posterior, float(log_z), steps, evaluations)


# Model adapters: this is where formulas, data layout and noise are kept.

def likelihood_target(hypothesis: Hypothesis, data: Dataset, *,
                      log_likelihood=None) -> ParticleTarget:
    """Adapt scalar log-density methods to the batched numerical interface.

    Uses hypothesis.log_prior(theta) and hypothesis.log_likelihood(data, theta).
    Alternatively supply log_likelihood(data, theta) explicitly, for a model
    whose likelihood is implemented elsewhere. The likelihood may use arbitrary
    named designs and vector observations; this adapter does not interpret them.

    Example for a program with an evaluable likelihood:
        target = likelihood_target(program, data)
        fit = tempered_smc(target, n_particles=2000, rng=rng)
    """
    likelihood = (getattr(hypothesis, "log_likelihood", None)
                  if log_likelihood is None else log_likelihood)
    if not callable(likelihood):
        raise ValueError("Supply an evaluable log_likelihood(data, theta); simulation alone is insufficient.")
    observed = tuple(data)

    def log_prior_batch(theta):
        return np.array([hypothesis.log_prior(t) for t in theta], dtype=float)

    def log_likelihood_batch(theta):
        return np.array([likelihood(observed, t) for t in theta], dtype=float)

    return ParticleTarget(hypothesis, log_prior_batch, log_likelihood_batch)


def polynomial_log_likelihood(hypothesis, data, sigma):
    """Batched log likelihood for polynomial means and known Gaussian noise.

    Build the design matrix once. Keep the Gaussian normalization constant:
    it cancels from some parameter updates, but NOT from model evidence.
    """
    if not isinstance(hypothesis, PolynomialHypothesis):
        raise ValueError("This polynomial adapter requires PolynomialHypothesis.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Noise SD must be positive and finite.")
    xs, ys = observed_arrays(data)
    X = design_matrix(hypothesis, xs)

    def log_likelihood(theta):
        residual = ys[None, :] - np.asarray(theta) @ X.T
        return -0.5*np.sum((residual/sigma)**2, axis=1) - len(ys)*np.log(sigma*np.sqrt(2*np.pi))

    return log_likelihood


def polynomial_target(hypothesis: PolynomialHypothesis, data: Dataset, sigma=None) -> ParticleTarget:
    """The polynomial special case: independent Gaussian coefficients and noise.

    sigma defaults to the hypothesis's observation SD. An explicit sigma makes
    a copy with matched observation noise, preserving earlier notebook calls.
    The vectorized prior and likelihood are an optimization, not part of SMC.
    """
    if not isinstance(hypothesis, PolynomialHypothesis):
        raise ValueError("This polynomial adapter requires PolynomialHypothesis.")
    sigma = hypothesis.noise.sigma if sigma is None else sigma
    hypothesis = observation_hypothesis(hypothesis, sigma)
    tau = hypothesis.prior_scale
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("Prior SD must be positive and finite.")

    def log_prior(theta):
        return (-0.5*np.sum((theta/tau)**2, axis=1)
                - hypothesis.n_params*np.log(tau*np.sqrt(2*np.pi)))

    return ParticleTarget(hypothesis, log_prior, polynomial_log_likelihood(hypothesis, data, sigma))


def polynomial_importance_fit(hypothesis, data, sigma=None, *, n_particles, rng) -> ParticleFit:
    """Notebook convenience: construct the polynomial target, then run general IS."""
    return importance_fit(polynomial_target(hypothesis, data, sigma), n_particles=n_particles, rng=rng)


def polynomial_tempered_smc(hypothesis, data, sigma=None, *, n_particles, rng, **options) -> ParticleFit:
    """Notebook convenience: construct the polynomial target, then run general SMC."""
    return tempered_smc(polynomial_target(hypothesis, data, sigma),
                        n_particles=n_particles, rng=rng, **options)


# Numerical helpers: keep validation, temperature choice and movement out of the main loop.

def _count(value, name, minimum=0):
    """Reject ambiguous counts (including booleans) before allocating particles."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _log_values(function, theta, name):
    """Require one valid log density per particle; zero density is allowed."""
    values = np.asarray(function(theta), float)
    if values.shape != (len(theta),) or np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError(f"{name} must return one log density per particle, finite or -inf.")
    return values


def _initial_particles(target, n, rng):
    """Draw from the declared prior and check both parameter and likelihood support."""
    if not isinstance(target, ParticleTarget):
        raise TypeError("Supply a ParticleTarget; use likelihood_target or polynomial_target.")
    _count(n, "n_particles", minimum=2)
    _count(target.hypothesis.n_params, "n_params", minimum=1)
    theta = np.asarray(target.hypothesis.sample_prior(rng, n), float)
    if theta.shape != (n, target.hypothesis.n_params) or not np.all(np.isfinite(theta)):
        raise ValueError("Prior samples must be finite with shape (n_particles, n_params).")
    prior = _log_values(target.log_prior, theta, "log_prior")
    if not np.all(np.isfinite(prior)):
        raise ValueError("Prior sampler produced draws outside its declared density support.")
    ell = _log_values(target.log_likelihood, theta, "log_likelihood")
    if not np.any(np.isfinite(ell)):
        raise ArithmeticError("All particles have zero likelihood; increase the population or inspect the model/data.")
    return theta, ell


def effective_sample_size(weights: np.ndarray) -> float:
    """ESS of normalized weights; not distinct particle count or proof of accuracy."""
    weights = np.asarray(weights, float)
    if (weights.ndim != 1 or not len(weights) or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0)):
        raise ValueError("ESS requires finite, nonnegative, normalized weights.")
    return float(1.0 / np.sum(weights**2))


def _likelihood_weights(log_weights):
    """Normalize weights and return log(mean(exp(log_weights))) stably."""
    normalizer = logsumexp(log_weights)
    if not np.isfinite(normalizer):
        raise ArithmeticError("No finite total importance weight.")
    return np.exp(log_weights-normalizer), float(normalizer-np.log(len(log_weights)))


def _temperature_schedule(temperatures):
    """Validate an optional fixed schedule before performing inference."""
    if temperatures is None:
        return None
    schedule = np.asarray(temperatures, float)
    if (schedule.ndim != 1 or len(schedule) < 2 or not np.all(np.isfinite(schedule))
            or schedule[0] != 0 or schedule[-1] != 1 or np.any(np.diff(schedule) <= 0)):
        raise ValueError("Temperatures must increase strictly from 0 to 1.")
    return schedule


def _next_temperature(beta, ell, ess_fraction):
    """Choose an increment by bisection on ESS.

    Zero-likelihood particles lose all weight at ANY positive temperature.
    Base the ESS goal on the finite-likelihood survivors so this unavoidable
    support change cannot stall the schedule. Their lost mass is still included
    in the evidence ratio, whose denominator is the FULL particle count.
    """
    goal = ess_fraction*np.count_nonzero(np.isfinite(ell))

    def ess_at(candidate):
        weights, _ = _likelihood_weights((candidate-beta)*ell)
        return effective_sample_size(weights)

    if ess_at(1.0) >= goal:
        return 1.0
    lo, hi = beta, 1.0
    for _ in range(50):
        midpoint = (lo+hi)/2
        if midpoint == beta:
            break
        if ess_at(midpoint) < goal:
            hi = midpoint
        else:
            lo = midpoint
    if lo <= beta:
        raise ArithmeticError("Temperature cannot advance at floating-point precision.")
    return lo


def _proposal_cholesky(theta, weights, prior_variance, scale):
    """Weighted-cloud covariance with a small empirical-prior diagonal floor."""
    centered = theta - weights @ theta
    covariance = (centered.T*weights) @ centered + np.diag(prior_variance*1e-8)
    return np.linalg.cholesky(covariance) * (scale*2.38/np.sqrt(theta.shape[1]))


def _metropolis_moves(target, theta, ell, beta, chol, moves, rng):
    """Symmetric random-walk Metropolis with the GENERAL prior density ratio.

    Reject out-of-prior proposals before calling the likelihood; do not clip
    or reflect them. For a symmetric proposal, the acceptance log ratio is
    log p(theta_new)-log p(theta_old) + beta*(log L_new-log L_old).
    Likelihood work counts only the proposals actually evaluated.
    """
    if moves == 0:
        return theta, ell, 0.0, 0
    prior = _log_values(target.log_prior, theta, "log_prior")
    accepted, evaluations = 0, 0
    for _ in range(moves):
        proposal = theta + rng.standard_normal(theta.shape) @ chol.T
        proposed_prior = _log_values(target.log_prior, proposal, "log_prior")
        supported = np.isfinite(proposed_prior)
        proposed_ell = np.full(len(theta), -np.inf)
        if np.any(supported):
            proposed_ell[supported] = _log_values(target.log_likelihood, proposal[supported], "log_likelihood")
        log_ratio = proposed_prior-prior + beta*(proposed_ell-ell)
        accept = np.log(rng.random(len(theta))) < np.minimum(0.0, log_ratio)
        theta[accept], ell[accept], prior[accept] = proposal[accept], proposed_ell[accept], proposed_prior[accept]
        accepted += int(accept.sum())
        evaluations += int(supported.sum())
    return theta, ell, accepted/(len(theta)*moves), evaluations


class ParticleUpdater(Updater):
    """Full-history inference for an existing candidate pool.

    target_factory(hypothesis, data) connects a model to the numerical engine.
    The default uses scalar prior/likelihood methods; a caller can supply other
    adapters without changing IS or SMC. last_fits exposes per-model diagnostics.

    Each call restarts from priors, NEVER multiplying the full history into an
    old posterior. Model weights use equal model priors and are conditional on
    this supplied pool. An outer discovery loop does the proposal, pruning and
    experiment choice. Latent-state filtering and online parameter updates are
    separate future components, not hidden in this updater.
    """
    def __init__(self, *, target_factory=likelihood_target, method="smc",
                 n_particles=1000, seed=0, **smc_options):
        if method not in {"smc", "importance"}:
            raise ValueError("method must be smc or importance.")
        if method == "importance" and smc_options:
            raise ValueError("SMC options do not apply to prior importance sampling.")
        if not callable(target_factory):
            raise ValueError("target_factory must be callable.")
        _count(n_particles, "n_particles", minimum=2)
        self.target_factory, self.method, self.n_particles = target_factory, method, n_particles
        self.rng = np.random.default_rng(seed)
        self.smc_options = smc_options
        self.last_fits = {}

    def update(self, data: Dataset, hypotheses: Sequence[Hypothesis]) -> Beliefs:
        """Fit parameters and evidence independently, then normalize within the pool."""
        names = [h.name for h in hypotheses]
        if not names or len(set(names)) != len(names):
            raise ValueError("Supply a nonempty pool with unique names.")
        fitter = tempered_smc if self.method == "smc" else importance_fit
        self.last_fits = {}
        fits = {h.name: fitter(self.target_factory(h, data), n_particles=self.n_particles,
                               rng=self.rng, **self.smc_options) for h in hypotheses}
        self.last_fits = fits
        log_z = {name: result.log_evidence for name, result in fits.items()}
        return Beliefs(self.method, "Finite-particle approximation, conditional on the supplied pool.",
                       {name: result.posterior for name, result in fits.items()},
                       model_probabilities(log_z), log_z, "bayesian_model_probability")


class PolynomialParticleUpdater(ParticleUpdater):
    """Explicit polynomial specialization; all inference remains in ParticleUpdater.

    A fixed sigma preserves the existing blog's known-noise setup. With sigma=None
    each polynomial uses its own declared observation noise instead.
    """
    def __init__(self, sigma=None, *, method="smc", n_particles=1000, seed=0, **smc_options):
        self.sigma = sigma
        super().__init__(target_factory=partial(polynomial_target, sigma=sigma),
                         method=method, n_particles=n_particles, seed=seed, **smc_options)
