from typing import NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp


class FastWeights(NamedTuple):
    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array


class FastState(NamedTuple):
    weights: FastWeights
    position: jax.Array


class TTTParams(NamedTuple):
    qkv: jax.Array
    gate: jax.Array
    inner_lr: jax.Array


class TTTContext(NamedTuple):
    params: TTTParams
    state: FastState
    update: jax.Array | bool = True


class TTT(nnx.Module):
    """Learned parameters for fast-weight memory in the final action-expert layers."""

    def __init__(self, layers: int, width: int, hidden_dim: int, *, rngs: nnx.Rngs):
        def kernel(shape):
            init = jax.vmap(nnx.initializers.lecun_uniform(), in_axes=(0, None))
            return nnx.Param(init(jax.random.split(rngs.params(), layers), shape))

        self.qkv = kernel((width, 3 * width))
        self.w1 = kernel((width, hidden_dim))
        self.b1 = nnx.Param(jnp.zeros((layers, hidden_dim)))
        self.w2 = kernel((hidden_dim, width))
        self.b2 = nnx.Param(jnp.zeros((layers, width)))
        self.gate = nnx.Param(jnp.full((layers, width), 0.001))
        self.log_inner_lr_multiplier = nnx.Param(jnp.zeros(layers))

    def initial_state(self, batch_size: int) -> FastState:
        weights = FastWeights(self.w1.value, self.b1.value, self.w2.value, self.b2.value)
        return FastState(
            jax.tree.map(
                lambda x: jnp.broadcast_to(x[:, None].astype(jnp.float32), (x.shape[0], batch_size, *x.shape[1:])),
                weights,
            ),
            jnp.array(0, dtype=jnp.int32),
        )

    def params(self) -> TTTParams:
        return TTTParams(self.qkv.value, self.gate.value, 0.1 * jnp.exp(self.log_inner_lr_multiplier.value))


def _fast_mlp(weights: FastWeights, x: jax.Array) -> jax.Array:
    hidden = jax.nn.gelu(x @ weights.w1 + weights.b1, approximate=False)
    return hidden @ weights.w2 + weights.b2


def _binding_loss(weights: FastWeights, keys: jax.Array, values: jax.Array) -> jax.Array:
    return jnp.mean(jnp.square(_fast_mlp(weights, keys) - values))


def _temporal_rope(x: jax.Array, positions: jax.Array) -> jax.Array:
    width = x.shape[-1]
    frequencies = 10_000.0 ** (-jnp.arange(0, width, 2, dtype=jnp.float32) / width)
    angles = positions[:, None, None, None] * frequencies
    even, odd = x[..., ::2], x[..., 1::2]
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    return jnp.stack((cos * even - sin * odd, sin * even + cos * odd), axis=-1).reshape(x.shape)


def apply_layer(
    tokens: jax.Array,
    params: TTTParams,
    weights: FastWeights,
    position: jax.Array,
    update: jax.Array | bool,
) -> tuple[jax.Array, FastWeights]:
    """Adapt each trajectory independently across the observation axis."""
    batch_size = weights.w1.shape[0]
    sequence = tokens.reshape(batch_size, -1, *tokens.shape[1:]).swapaxes(0, 1)
    q, k, v = jnp.split(sequence.astype(jnp.float32) @ params.qkv.astype(jnp.float32), 3, axis=-1)
    positions = position + jnp.arange(sequence.shape[0])
    q, k = _temporal_rope(q, positions), _temporal_rope(k, positions)

    def step(weights, inputs):
        queries, keys, values = inputs

        def adapt(weights):
            gradients = jax.vmap(jax.grad(_binding_loss))(weights, keys, values)
            return jax.tree.map(lambda weight, grad: weight - params.inner_lr * grad, weights, gradients)

        weights = jax.lax.cond(update, adapt, lambda weights: weights, weights)
        return weights, jax.vmap(_fast_mlp)(weights, queries)

    weights, outputs = jax.lax.scan(step, weights, (q, k, v))
    residual = outputs.swapaxes(0, 1).reshape(tokens.shape) * jnp.tanh(params.gate.astype(jnp.float32))
    return tokens + residual.astype(tokens.dtype), weights
