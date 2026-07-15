from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.networks import ChunkActor
from rlt_online_rl.networks import TwinCritic
from rlt_online_rl.networks import apply_reference_dropout
from rlt_online_rl.networks import compute_actor_loss
from rlt_online_rl.networks import compute_critic_loss


def _config() -> RLTOnlineRLConfig:
    return RLTOnlineRLConfig(
        action_dim=3,
        chunk_len=4,
        z_dim=5,
        proprio_dim=2,
        actor_hidden_dim=32,
        critic_hidden_dim=32,
        actor_num_layers=2,
        critic_num_layers=2,
    )


def test_actor_output_shape() -> None:
    cfg = _config()
    actor = ChunkActor(cfg.z_dim, cfg.proprio_dim, cfg.chunk_len, cfg.action_dim, 32, 2, cfg.fixed_std)
    params = actor.init_params(jax.random.PRNGKey(0))
    z = jnp.ones((6, cfg.z_dim))
    proprio = jnp.ones((6, cfg.proprio_dim))
    ref = jnp.ones((6, cfg.chunk_len, cfg.action_dim))
    mu = actor.actor_mean(params, z, proprio, ref)
    assert mu.shape == (6, cfg.chunk_len, cfg.action_dim)


def test_twin_critic_output_shape() -> None:
    cfg = _config()
    critic = TwinCritic(cfg.z_dim, cfg.proprio_dim, cfg.chunk_len, cfg.action_dim, 32, 2)
    params = critic.init_params(jax.random.PRNGKey(1))
    z = jnp.ones((6, cfg.z_dim))
    proprio = jnp.ones((6, cfg.proprio_dim))
    action = jnp.ones((6, cfg.chunk_len, cfg.action_dim))
    q1, q2 = critic.q_values(params, z, proprio, action)
    assert q1.shape == (6,)
    assert q2.shape == (6,)


def test_nero_networks_use_19d_actions_with_complete_26d_proprio() -> None:
    cfg = RLTOnlineRLConfig(
        action_dim=19,
        chunk_len=10,
        z_dim=8,
        proprio_dim=26,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        actor_num_layers=1,
        critic_num_layers=1,
    )
    actor = ChunkActor(
        cfg.z_dim,
        cfg.proprio_dim,
        cfg.chunk_len,
        cfg.action_dim,
        cfg.actor_hidden_dim,
        cfg.actor_num_layers,
        cfg.fixed_std,
    )
    critic = TwinCritic(
        cfg.z_dim,
        cfg.proprio_dim,
        cfg.chunk_len,
        cfg.action_dim,
        cfg.critic_hidden_dim,
        cfg.critic_num_layers,
    )
    actor_params = actor.init_params(jax.random.PRNGKey(10))
    critic_params = critic.init_params(jax.random.PRNGKey(11))
    z_rl = jnp.ones((2, cfg.z_dim))
    proprio = jnp.ones((2, cfg.proprio_dim))
    ref_chunk = jnp.ones((2, cfg.chunk_len, cfg.action_dim))

    action_chunk = actor.actor_mean(actor_params, z_rl, proprio, ref_chunk)
    q1, q2 = critic.q_values(critic_params, z_rl, proprio, action_chunk)

    assert action_chunk.shape == (2, 10, 19)
    assert q1.shape == (2,)
    assert q2.shape == (2,)


def test_reference_dropout_zeroes_entire_chunk() -> None:
    ref_chunk = jnp.ones((8, 4, 3))
    dropped = apply_reference_dropout(jax.random.PRNGKey(2), ref_chunk, 1.0)
    assert jnp.allclose(dropped, 0.0)


def test_actor_and_critic_losses_are_scalars() -> None:
    cfg = _config()
    actor = ChunkActor(cfg.z_dim, cfg.proprio_dim, cfg.chunk_len, cfg.action_dim, 32, 2, cfg.fixed_std)
    critic = TwinCritic(cfg.z_dim, cfg.proprio_dim, cfg.chunk_len, cfg.action_dim, 32, 2)
    actor_params = actor.init_params(jax.random.PRNGKey(3))
    critic_params = critic.init_params(jax.random.PRNGKey(4))
    batch_size = 5
    z = jnp.ones((batch_size, cfg.z_dim))
    proprio = jnp.ones((batch_size, cfg.proprio_dim))
    ref = jnp.ones((batch_size, cfg.chunk_len, cfg.action_dim))
    action = jnp.ones((batch_size, cfg.chunk_len, cfg.action_dim))
    rewards = jnp.ones((batch_size, cfg.chunk_len))
    done = jnp.zeros((batch_size,))
    actor_loss, _ = compute_actor_loss(
        actor,
        actor_params,
        critic,
        critic_params,
        z,
        proprio,
        ref,
        1.0,
        cfg.reference_dropout_prob,
        jax.random.PRNGKey(5),
    )
    critic_loss, _ = compute_critic_loss(
        critic,
        critic_params,
        actor,
        actor_params,
        critic_params,
        z,
        proprio,
        action,
        rewards,
        done,
        z,
        proprio,
        ref,
        cfg.gamma,
        jax.random.PRNGKey(6),
    )
    assert actor_loss.shape == ()
    assert critic_loss.shape == ()
