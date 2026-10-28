"""Generate three- and six-component fields on the same nonperiodic 3D graph.

Run ``python demo_vector.py`` from this checkout. Only JAX is required.
"""

from functools import partial

import jax
import jax.numpy as jnp

from graphgp import build_graph
from graphgp.refine_vector import generate, generate_inv, generate_logdet


def make_covariance(m, *, spatial_mixing):
    """Mix independent latent RBF kernels into an (m, m) covariance."""
    channels = jnp.arange(m)
    lengths = 0.22 + 0.03 * channels if spatial_mixing else jnp.full(m, 0.22)
    eye = jnp.eye(m)

    def mixing(point):
        if spatial_mixing:
            return eye + 0.08 * jnp.outer(jnp.sin(point[0] + channels), jnp.cos(point[2] + 0.3 * channels))
        return eye + 0.12 * jnp.ones((m, m))

    def covariance(x, y):
        latent = jnp.exp(-jnp.sum((x - y) ** 2) / (2 * lengths**2))
        # K(x,y) = A(x) diag(k_q(x,y)) A(y).T is a valid covariance.
        block = (mixing(x) * latent) @ mixing(y).T
        return block + 1e-5 * eye * jnp.all(x == y)

    return covariance


def main():
    jax.config.update("jax_enable_x64", True)
    axis = jnp.linspace(0.0, 1.0, 5)
    points = jnp.stack(jnp.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    graph = build_graph(points, n0=8, k=4)

    for m in (3, 6):
        covariance = make_covariance(m, spatial_mixing=m == 6)
        xi = jax.random.normal(jax.random.key(m), (len(points), m))
        field = jax.jit(partial(generate, graph, covariance))(xi)
        recovered = generate_inv(graph, covariance, field)
        error = float(jnp.max(jnp.abs(recovered - xi)))
        assert bool(jnp.all(jnp.isfinite(field))) and error < 1e-8
        logdet = float(generate_logdet(graph, covariance))
        print(f"m={m}: field shape={field.shape}, inverse max error={error:.2e}, logdet={logdet:.3f}")

    # The six-component case is nonstationary and nonseparable. Components
    # are correlated channels at each 3D point, not extra spatial axes.
    # These checks concern the graph approximation, not the exact dense GP.


if __name__ == "__main__":
    main()
