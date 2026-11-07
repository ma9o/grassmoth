"""mGRAD with retained proposal labels and marginalized Gaussian u.

Extend the physical path target by independent labels with law r, the corrected
parameter prior. Free particles choose components with running-prefix law alpha;
the reference label is refreshed from r. For candidate n the bank weight is
Q_n r(b_n) prod_{i != n} alpha_i(b_i) times the integrated Gaussian bank density.
Dropping the common product of alpha gives log r(b_n) - log alpha_n(b_n).
The labels are auxiliary to the path and distinct from MPGibbs' parameter label.

Corenflos & Finke (2024), https://arxiv.org/abs/2401.14868v1, Appendix C.2,
Lemma 3 gives the heterogeneous Gaussian integral. Appendix D.2's bank swap
identity applies to this extended target; r is independent of
the path, so it cancels in backward selection. This is a retained-component
extension, not the fully component-marginalized mixture kernel. Early prefix
collapse can cause large label corrections and poor mixing.
"""

from functools import partial
from typing import Annotated, NamedTuple, cast

import jax
import jax.numpy as jnp
from pydantic import Field

from grassmoth.models.ssm.inference.samplers.particle.guided_geometry import (
    _DiagonalProposalGeometry,
    _ProposalGeometry,
)
from grassmoth.models.ssm.inference.samplers.particle.math import (
    TRITON_NUM_WARPS,
    categorical_from_uniform,
)

GAUSSIAN_TILE_ELEMENTS: Annotated[
    int,
    Field(
        description=(
            "Heuristic: target at most 4096 matrix elements per Gaussian elimination tile, "
            "reducing the particle tile as the latent dimension grows. Each tile retains "
            "a complete matrix; this controls register use without splitting a factorization."
        )
    ),
] = 4096

GAUSSIAN_COORDINATE_TILE: Annotated[
    int,
    Field(
        description="Heuristic: eight independent coordinates per Gaussian bank tile bound register use while sharing particle reductions."
    ),
] = 8


def _gaussian_integral_weights(
    precision: jax.Array,
    shift: jax.Array,
    constant: jax.Array,
) -> jax.Array:
    """Fuse exact SPD elimination, the solved quadratic and log determinant.

    Successive Schur complements compute h' A^-1 h / 2 - log(det(A)) / 2
    without materializing a Cholesky factor or dispatching a triangular solve.
    The complete Gaussian correction uses CF24 Appendix C.2, Lemma 3. Matrix
    and particle tiles are specialized from static shapes, independently of the
    model's drift, observations or parameter values.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    count, dimension = shift.shape
    block = pl.next_power_of_2(dimension)
    particle_block = min(pl.next_power_of_2(count), max(1, GAUSSIAN_TILE_ELEMENTS // block**2))

    def kernel(precision_ref, shift_ref, constant_ref, out_ref):
        p = pl.program_id(0) * particle_block + jnp.arange(particle_block)
        d = jnp.arange(block)
        valid = p < count
        matrix = plt.load(
            precision_ref.at[p[:, None, None], d[None, :, None], d[None, None, :]],
            mask=valid[:, None, None]
            & (d[None, :, None] < dimension)
            & (d[None, None, :] < dimension),
            other=0.0,
        )
        matrix = jnp.where(valid[:, None, None], matrix, d[:, None] == d[None, :])
        rhs = plt.load(
            shift_ref.at[p[:, None], d[None, :]],
            mask=valid[:, None] & (d[None, :] < dimension),
            other=0.0,
        )
        result = plt.load(constant_ref.at[p], mask=valid, other=0.0)

        def eliminate(index, carry):
            matrix, rhs, result = carry
            column = jnp.sum(jnp.where(d[None, None, :] == index, matrix, 0.0), axis=-1)
            pivot = jnp.sqrt(jnp.sum(jnp.where(d[None, :] == index, column, 0.0), axis=-1))
            column = column / pivot[:, None]
            solved = jnp.sum(jnp.where(d[None, :] == index, rhs, 0.0), axis=-1) / pivot
            return (
                matrix - column[:, :, None] * column[:, None, :],
                rhs - column * solved[:, None],
                result + 0.5 * solved * solved - jnp.log(pivot),
            )

        _, _, result = jax.lax.fori_loop(0, dimension, eliminate, (matrix, rhs, result))
        plt.store(out_ref.at[p], result, mask=valid)

    def run(*args, interpret):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((count,), precision.dtype),
            grid=(pl.cdiv(count, particle_block),),
            interpret=interpret,
            compiler_params=plt.CompilerParams(num_warps=TRITON_NUM_WARPS),
            name="mgrad_gaussian_integral_weights",
        )(*args)

    return cast(
        "jax.Array",
        jax.lax.platform_dependent(
            precision,
            shift,
            constant,
            cpu=partial(run, interpret=True),
            cuda=partial(run, interpret=False),
        ),
    )


def _diagonal_integral_terms(
    x: jax.Array,
    prior: jax.Array,
    chol: jax.Array,
    white: jax.Array,
    gradient: jax.Array,
    variance: jax.Array,
    free: jax.Array,
    valid: jax.Array,
    count: int,
) -> jax.Array:
    """CF24 C.2 coordinate terms with one shared auxiliary recentering per bank."""
    auxiliary = jnp.where(free & valid, x + variance * gradient, 0.0)
    origin = jnp.sum(auxiliary, axis=0, keepdims=True) / count
    auxiliary_map = chol * chol * (free / variance)
    residual = jnp.where(free, x - prior, 0.0) - auxiliary_map * origin
    whitened = white * residual
    mapped = white * auxiliary_map
    information = mapped * mapped
    shifts = mapped * whitened
    precision = (
        1.0 / variance
        + jnp.sum(jnp.where(valid, information, 0.0), axis=0, keepdims=True)
        - information
    )
    centered = auxiliary - origin
    shift = (
        centered / variance + jnp.sum(jnp.where(valid, shifts, 0.0), axis=0, keepdims=True) - shifts
    )
    factor = jnp.sqrt(precision)
    solved = shift / factor
    return (
        jnp.log(chol)
        - jnp.log(factor)
        + 0.5 * (whitened * whitened - centered * centered / variance + solved * solved)
    )


def _diagonal_mixture_weights(
    particles: jax.Array,
    prior_centers: jax.Array,
    cholesky: jax.Array,
    whitener: jax.Array,
    label_correction: jax.Array,
    gradients: jax.Array,
    variance: jax.Array,
    mask: jax.Array,
    *,
    interpret: bool,
) -> jax.Array:
    """Fuse the diagonal heterogeneous Gaussian integral with its bank reductions.

    Independence factorizes the same integral coordinate by coordinate. Each
    tile retains its particle reductions; their coordinate sums are accumulated
    once per candidate, with the auxiliary-label correction included once.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    count, dimension = particles.shape
    block = pl.next_power_of_2(count)
    coordinate_block = min(pl.next_power_of_2(dimension), GAUSSIAN_COORDINATE_TILE)
    tiles = pl.cdiv(dimension, coordinate_block)

    def kernel(
        x_ref, prior_ref, chol_ref, white_ref, label_ref, grad_ref, var_ref, mask_ref, out_ref
    ):
        p = jnp.arange(block)[:, None]
        d = (pl.program_id(0) * coordinate_block + jnp.arange(coordinate_block))[None, :]
        valid = (p < count) & (d < dimension)
        x = plt.load(x_ref.at[p, d], mask=valid, other=0.0)
        prior = plt.load(prior_ref.at[p, d], mask=valid, other=0.0)
        chol = plt.load(chol_ref.at[p, d], mask=valid, other=1.0)
        white = plt.load(white_ref.at[p, d], mask=valid, other=1.0)
        gradient = plt.load(grad_ref.at[p, d], mask=valid, other=0.0)
        variance = plt.load(var_ref.at[d], mask=d < dimension, other=1.0)
        free = plt.load(mask_ref.at[d], mask=d < dimension, other=False)
        result = _diagonal_integral_terms(
            x, prior, chol, white, gradient, variance, free, valid, count
        )
        correction = plt.load(label_ref.at[p[:, 0]], mask=p[:, 0] < count, other=0.0)
        result = jnp.sum(result, axis=1) + jnp.where(pl.program_id(0) == 0, correction, 0.0)
        plt.store(out_ref.at[p[:, 0], pl.program_id(0)], result, mask=p[:, 0] < count)

    terms = cast(
        "jax.Array",
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((count, tiles), particles.dtype),
            grid=(tiles,),
            interpret=interpret,
            compiler_params=plt.CompilerParams(num_warps=TRITON_NUM_WARPS),
            name="diagonal_mgrad_mixture_weights",
        )(
            particles,
            prior_centers,
            cholesky,
            whitener,
            label_correction,
            gradients,
            variance,
            mask,
        ),
    )
    return jnp.sum(terms, axis=-1)


class MarginalMixtureBank(NamedTuple):
    """Selected component geometry and its exact auxiliary-label correction."""

    particles: jax.Array
    centers: jax.Array
    prior_centers: jax.Array
    geometry: _ProposalGeometry
    label_correction: jax.Array

    def log_weights(self, gradients: jax.Array, variance: jax.Array, mask: jax.Array) -> jax.Array:
        """Integrate u with heterogeneous covariance using leave-one-out information sums.

        Centre u at the bank's mean auxiliary centre before forming quadratics.
        Terms common to every candidate are discarded only after the Gaussian
        integral, including the sum of component log determinants and residuals.
        """
        diagonal = isinstance(self.geometry, _DiagonalProposalGeometry)
        if diagonal or self.particles.shape[-1] == 1:
            return cast(
                "jax.Array",
                jax.lax.platform_dependent(
                    self.particles,
                    self.prior_centers,
                    self.geometry.chol if diagonal else self.geometry.chol[..., 0],
                    self.geometry.whiten if diagonal else self.geometry.whiten[..., 0],
                    self.label_correction,
                    gradients,
                    variance,
                    mask,
                    cpu=partial(_diagonal_mixture_weights, interpret=True),
                    cuda=partial(_diagonal_mixture_weights, interpret=False),
                ),
            )
        covariance = self.geometry.chol @ jnp.swapaxes(self.geometry.chol, -1, -2)
        auxiliary_map = covariance * (mask / variance)[None, None, :]
        auxiliary_means = jnp.where(mask, self.particles + variance * gradients, 0.0)
        origin = jnp.mean(auxiliary_means, axis=0)
        residual = jnp.where(mask, self.particles - self.prior_centers, 0.0)
        residual = residual - jnp.einsum("pij,j->pi", auxiliary_map, origin)
        whitened = jnp.einsum("pij,pj->pi", self.geometry.whiten, residual)
        maps = self.geometry.whiten @ auxiliary_map
        information = jnp.swapaxes(maps, -1, -2) @ maps
        shifts = jnp.einsum("pji,pj->pi", maps, whitened)
        precision = jnp.diag(1.0 / variance) + jnp.sum(information, axis=0) - information
        centered_means = auxiliary_means - origin
        shift = centered_means / variance + jnp.sum(shifts, axis=0) - shifts
        component_determinant = jnp.sum(
            jnp.log(jnp.diagonal(self.geometry.chol, axis1=-2, axis2=-1)), axis=-1
        )
        constant = (
            self.label_correction
            + component_determinant
            + 0.5 * (jnp.sum(whitened**2, axis=-1) - jnp.sum(centered_means**2 / variance, axis=-1))
        )
        return _gaussian_integral_weights(precision, shift, constant)


def marginal_mixture_bank(
    proposal: _ProposalGeometry,
    centers: jax.Array,
    label_log_probs: jax.Array,
    reference_label_log_probs: jax.Array,
    noise: jax.Array,
    component_noise: jax.Array,
    reference: jax.Array,
    mask: jax.Array,
) -> MarginalMixtureBank:
    """Refresh the reference component from r and draw free components from alpha."""
    probabilities = label_log_probs.at[0].set(reference_label_log_probs)
    labels = categorical_from_uniform(probabilities, component_noise)
    return mixture_bank_from_labels(
        proposal,
        centers,
        label_log_probs,
        reference_label_log_probs,
        noise,
        labels,
        reference,
        mask,
    )


def mixture_bank_from_labels(
    proposal: _ProposalGeometry,
    centers: jax.Array,
    label_log_probs: jax.Array,
    reference_label_log_probs: jax.Array,
    noise: jax.Array,
    labels: jax.Array,
    reference: jax.Array,
    mask: jax.Array,
) -> MarginalMixtureBank:
    """Build the Gaussian bank from its refreshed labels and retain their correction.

    The reference label is sampled from its prior, but its correction still uses
    the carried parent score, exactly as for the other particles.
    """
    indices = jnp.arange(centers.shape[0])
    geometry = jax.tree.map(lambda value: value[labels], proposal)
    selected = centers[indices, labels]
    particles = jnp.where(
        mask & (indices[:, None] != 0),
        selected + geometry.transform_noise(jnp.pad(noise, ((1, 0), (0, 0)))),
        reference,
    )
    return MarginalMixtureBank(
        particles,
        selected,
        selected - geometry.shift,
        geometry,
        reference_label_log_probs[labels] - label_log_probs[indices, labels],
    )
