import jax.numpy as jnp
from jaxrl.value_learning import generalized_advantage_estimate


def test_gae():
    gamma = 0.9
    lmbda = 0.9

    values = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    rewards = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    terminated = jnp.array([0, 0, 0, 0, 1], dtype=jnp.int32)
    truncated = jnp.array([0, 0, 0, 0, 1], dtype=jnp.int32)
    advantage, value_target = generalized_advantage_estimate(
        gamma, lmbda, rewards, values, jnp.array(0.0), terminated, truncated
    )

    expected_advantage = jnp.array(
        [0.95116055, 0.92204994, 0.76500005, 0.45000002, 0.0]
    )
    expected_value_target = jnp.array([0.9740, 1.0568, 1.0245, 0.8500, 0.5000])

    assert jnp.allclose(advantage, expected_advantage, atol=1e-4)
    assert jnp.allclose(value_target, expected_value_target, atol=1e-4)


if __name__ == "__main__":
    test_gae()
