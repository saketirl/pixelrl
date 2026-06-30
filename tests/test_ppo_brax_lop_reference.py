import jax
import jax.numpy as jnp

import ppo_brax


def test_lop_reference_actor_and_critic_use_two_256_hidden_layers():
    obs = jnp.zeros((1, 27))
    action_dim = 8
    key = jax.random.PRNGKey(0)
    actor_key, critic_key = jax.random.split(key)

    actor = ppo_brax.LopReferenceActor(action_dim=action_dim, activation="relu")
    critic = ppo_brax.LopReferenceCritic(activation="relu")

    actor_params = actor.init(actor_key, obs)["params"]
    critic_params = critic.init(critic_key, obs)["params"]

    assert actor_params["Dense_0"]["kernel"].shape == (27, 256)
    assert actor_params["Dense_1"]["kernel"].shape == (256, 256)
    assert actor_params["Dense_2"]["kernel"].shape == (256, action_dim)
    assert actor_params["log_std"].shape == (action_dim,)

    assert critic_params["Dense_0"]["kernel"].shape == (27, 256)
    assert critic_params["Dense_1"]["kernel"].shape == (256, 256)
    assert critic_params["Dense_2"]["kernel"].shape == (256, 1)

    actor_mean, actor_logstd = actor.apply({"params": actor_params}, obs)
    value = critic.apply({"params": critic_params}, obs)

    assert actor_mean.shape == (1, action_dim)
    assert actor_logstd.shape == (action_dim,)
    assert value.shape == (1, 1)
