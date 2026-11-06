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

from typing import NamedTuple

import jax
import jax.numpy as jnp

from grassmoth.models.ssm.inference.samplers.particle.guided_geometry import _ProposalGeometry


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
        chol = jnp.linalg.cholesky(precision)
        solved = jax.vmap(lambda c, h: jax.scipy.linalg.solve_triangular(c, h, lower=True))(
            chol, shift
        )
        log_determinant = jnp.sum(jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)
        component_determinant = jnp.sum(
            jnp.log(jnp.diagonal(self.geometry.chol, axis1=-2, axis2=-1)), axis=-1
        )
        return (
            self.label_correction
            + component_determinant
            - log_determinant
            + 0.5
            * (
                jnp.sum(whitened**2, axis=-1)
                - jnp.sum(centered_means**2 / variance, axis=-1)
                + jnp.sum(solved**2, axis=-1)
            )
        )


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
    labels = jnp.argmax(probabilities + component_noise, axis=-1)
    indices = jnp.arange(centers.shape[0])
    geometry = jax.tree.map(lambda value: value[labels], proposal)
    selected = centers[indices, labels]
    free = jnp.where(
        mask,
        selected[1:] + jnp.einsum("pij,pj->pi", geometry.chol[1:], noise),
        reference,
    )
    return MarginalMixtureBank(
        jnp.concatenate((reference[None], free)),
        selected,
        selected - geometry.shift,
        geometry,
        reference_label_log_probs[labels] - label_log_probs[indices, labels],
    )
