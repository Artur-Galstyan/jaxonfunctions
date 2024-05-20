import jax
import jax.numpy as jnp
from beartype.typing import Optional
from jaxtyping import Array, Float, Int


def sarsa(
    q_tm1: Float[Array, " n_actions"],
    a_tm1: Int[Array, ""] | int,
    r_t: Float[Array, ""] | float,
    discount_t: Float[Array, ""] | float,
    q_t: Float[Array, " n_actions"],
    a_t: Int[Array, ""] | int,
    stop_target_gradients: bool = True,
) -> Float[Array, ""]:
    """Calculates the SARSA temporal difference error.

    See "Reinforcement Learning: An Introduction" by Sutton and Barto.
    (http://incompleteideas.net/book/ebook/node64.html.)

    Args:
        q_tm1: Float[Array, " n_actions"] Q-values at time t-1.
        a_tm1: Int[Array, ""] action index at time t-1.
        r_t: Float[Array, ""] reward at time t.
        discount_t: Float[Array, ""] discount factor at time t.
        q_t: Float[Array, " n_actions"] Q-values at time t.
        a_t: Int action index at time t.
        stop_target_gradients: bool indicating whether or not to apply stop gradient to
        targets.
    Returns:
      SARSA temporal difference error.
    """
    target_tm1 = r_t + discount_t * q_t[a_t]
    target_tm1 = jax.lax.select(
        stop_target_gradients, jax.lax.stop_gradient(target_tm1), target_tm1
    )
    return target_tm1 - q_tm1[a_tm1]


def q_learning(
    q_tm1: Float[Array, " n_actions"],
    a_tm1: Int[Array, ""] | int,
    r_t: Float[Array, ""] | float,
    discount_t: Float[Array, ""] | float,
    q_t: Float[Array, " n_actions"],
    stop_target_gradients: bool = True,
):
    """Calculates the Q-learning temporal difference error.

    See "Reinforcement Learning: An Introduction" by Sutton and Barto.
    (http://incompleteideas.net/book/ebook/node65.html.)

    Args:
        q_tm1: Float[Array, " n_actions"] Q-values at time t-1.
        a_tm1: Int[Array, ""] action index at time t-1.
        r_t: Float[Array, ""] reward at time t.
        discount_t: Float[Array, ""] discount factor at time t.
        q_t: Float[Array, " n_actions"] Q-values at time t.
        stop_target_gradients: bool indicating whether or not to apply stop gradient to
        targets.
    Returns:
        Q-learning temporal difference error.
    """

    target_tm1 = r_t + discount_t * jnp.max(q_t)
    target_tm1 = jax.lax.select(
        stop_target_gradients, jax.lax.stop_gradient(target_tm1), target_tm1
    )

    return target_tm1 - q_tm1[a_tm1]


def generalized_advantage_estimate(
    gamma: Float[Array, ""] | float,
    lmbda: Float[Array, ""] | float,
    rewards: Float[Array, " n_steps"],
    values: Float[Array, " n_steps"],
    bootstrap_value: Float[Array, ""],
    terminated: Int[Array, " n_steps"],
    truncated: Optional[Int[Array, " n_steps"]] = None,
    stop_target_gradients: bool = True,
):
    if truncated is None:
        truncated = terminated
    truncation_mask = 1 - truncated
    values_t_plus_1 = jnp.concatenate(
        [values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0
    )

    deltas = (
        rewards + (gamma * (1 - terminated)) * values_t_plus_1 - values
    ) * truncation_mask
    discounts = gamma * lmbda * truncation_mask

    def body(carry, x):
        delta, discount = x
        carry = delta + discount * carry
        return carry, carry

    _, vs_minus_v_xs = jax.lax.scan(body, 0.0, xs=(deltas, discounts), reverse=True)
    value_target = vs_minus_v_xs + values

    vs_t_plus_1 = jnp.concatenate(
        [value_target[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0
    )

    advantage = (
        rewards + (gamma * (1 - terminated)) * vs_t_plus_1 - values
    ) * truncation_mask

    if stop_target_gradients:
        return jax.lax.stop_gradient(advantage), jax.lax.stop_gradient(value_target)
    else:
        return advantage, value_target
