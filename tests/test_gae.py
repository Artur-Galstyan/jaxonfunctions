import jax.numpy as jnp
from jaxrl.value_learning import generalized_advantage_estimate


def test_gae():
    gamma = 0.9
    lmbda = 0.9

    state_value = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    rewards = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    next_state_value = jnp.array([0.2, 0.3, 0.4, 0.5, 0.0])
    dones = jnp.array([0, 0, 0, 0, 1], dtype=jnp.int32)

    advantage, value_target = generalized_advantage_estimate(
        gamma, lmbda, rewards, state_value, next_state_value, dones
    )

    expected_advantage = jnp.array([0.8740, 0.8568, 0.7245, 0.4500, 0.0000])
    expected_value_target = jnp.array([0.9740, 1.0568, 1.0245, 0.8500, 0.5000])

    assert jnp.allclose(advantage, expected_advantage, atol=1e-4)
    assert jnp.allclose(value_target, expected_value_target, atol=1e-4)


if __name__ == "__main__":
    test_gae()
