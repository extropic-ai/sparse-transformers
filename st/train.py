import equinox as eqx
import optax
from jax import numpy as jnp


@eqx.filter_jit
def loss(model, data_batch, tokenizer):
    x, y = data_batch[:, :-1], data_batch[:, 1:]
    logits = eqx.filter_vmap(model, in_axes=(0, None))(x, None)
    train_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    mask = y != tokenizer.encoder["<PAD>"]
    train_loss = train_loss * mask
    return jnp.sum(train_loss) / jnp.sum(mask)


@eqx.filter_jit
def step(
    model,
    data_batch,
    optimizer,
    opt_state,
    tokenizer,
    model_shardings=None,
    opt_shardings=None,
):
    if model_shardings is not None:
        model = eqx.filter_shard(model, model_shardings)
        opt_state = eqx.filter_shard(opt_state, opt_shardings)
    value, grads = eqx.filter_value_and_grad(loss)(model, data_batch, tokenizer)
    if model_shardings is not None:
        grads = eqx.filter_shard(grads, model_shardings)
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
    )
    model = eqx.apply_updates(model, updates)
    if model_shardings is not None:
        model = eqx.filter_shard(model, model_shardings)
        opt_state = eqx.filter_shard(opt_state, opt_shardings)
    return model, value, opt_state
