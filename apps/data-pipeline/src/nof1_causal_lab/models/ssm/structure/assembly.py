"""Single source of truth for block-owned sparse assembly algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np


def dense_vector_positions(mask: np.ndarray | jnp.ndarray, n: int) -> list[int]:
    """``[idx]`` positions marked free on a 1-D length-``n`` mask."""
    return [idx for idx in range(n) if bool(mask[idx])]


def rect_matrix_positions(
    mask: np.ndarray | jnp.ndarray, n_rows: int, n_cols: int
) -> list[tuple[int, int]]:
    """``(row, col)`` positions marked free on a rectangular mask."""
    return [(i, j) for i in range(n_rows) for j in range(n_cols) if bool(mask[i, j])]


def chol_diag_positions(mask: np.ndarray | jnp.ndarray, n: int) -> list[int]:
    """Diagonal positions marked free on a square Cholesky-shape mask."""
    return [idx for idx in range(n) if bool(mask[idx, idx])]


def strict_lower_positions(mask: np.ndarray | jnp.ndarray, n: int) -> list[tuple[int, int]]:
    """Strict-lower-triangle ``(row, col)`` positions marked free."""
    return [(row, col) for row in range(n) for col in range(row) if bool(mask[row, col])]


# ---------------------------------------------------------------------------
# Generic sparse-substitution helpers (used by the BlockSpec abstraction
# in ``blocks.py``)
# ---------------------------------------------------------------------------


def assemble_sparse[Position: int | tuple[int, int]](
    template: jnp.ndarray,
    free_positions: tuple[Position, ...],
    free: jnp.ndarray | None,
) -> jnp.ndarray:
    """Insert free values at typed vector or matrix coordinates."""
    out = jnp.asarray(template)
    if free is not None:
        free = jnp.asarray(free, dtype=out.dtype)
        for index, position in enumerate(free_positions):
            out = out.at[position].set(free[index])
    return out


def assemble_diffusion_chol(
    diffusion_chol_template: jnp.ndarray,
    diag_positions: Sequence[int],
    lower_positions: Sequence[tuple[int, int]],
    diag_free: jnp.ndarray | None,
    lower_free: jnp.ndarray | None,
) -> jnp.ndarray:
    """Process-noise Cholesky assembler shared by ``DiffusionBlockSpec`` and ``DynamicalModelSpec``."""
    diffusion = jnp.asarray(diffusion_chol_template)
    if diag_free is not None:
        diag_free = jnp.asarray(diag_free, dtype=diffusion.dtype)
        for idx, latent_idx in enumerate(diag_positions):
            diffusion = diffusion.at[latent_idx, latent_idx].set(diag_free[idx])
    if lower_free is not None:
        lower_free = jnp.asarray(lower_free, dtype=diffusion.dtype)
        for idx, (row, col) in enumerate(lower_positions):
            diffusion = diffusion.at[row, col].set(lower_free[idx])
    return diffusion


def assemble_manifest_chol(
    template: jnp.ndarray,
    free_positions: Sequence[int],
    free: jnp.ndarray | None,
) -> jnp.ndarray:
    """Insert free diagonal observation-noise scales while retaining fixed template entries."""
    chol = jnp.asarray(template)
    if free is not None and len(free_positions) > 0:
        free = jnp.asarray(free, dtype=chol.dtype)
        for idx, manifest_idx in enumerate(free_positions):
            chol = chol.at[manifest_idx, manifest_idx].set(free[idx])
    return chol
