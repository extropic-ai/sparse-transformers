"""Text-scaling experiment for z1t — the AFT sibling of `text_scaling.py`.

Same experiment and the same wandb keys; the model is a z1t AFT transformer
rather than an st `Transformer`, so the parameter sharding follows z1t's tree
(embedding / pe / blocks[i].attn / blocks[i].mlp / clf) instead of st's.

    python -m experiments.z1t_scaling --config configs/z1t_tiny.yaml

Two deliberate differences from `text_scaling.py`:

- No FLOPs accounting. `st.flops` is derived for windowed softmax attention and
  has no AFT counterpart, so there is no `total_flops` / `target_flops` here.
- Batches come from z1t's own loaders, which already return `(inp, tgt)` shifted
  by one, so there is no PAD mask: every z1t corpus is a contiguous token
  stream or a fixed-length sample.

What this adds over `z1t.train` is real parameter sharding — `z1t.train` only
shards the batch (`training.ddp`), while this shards the parameters too.
"""

import argparse
from dataclasses import asdict

import equinox as eqx
import jax
import numpy as np
import optax
import wandb
from jax.sharding import NamedSharding, PartitionSpec
from tqdm import tqdm

from z1t.components import active_params
from z1t.model import create_model
from z1t.train import config_path, load_config, make_data, model_config, run_name


@eqx.filter_jit
def loss(model, x, y):
    logits = jax.vmap(model)(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


@eqx.filter_jit
def step(model, x, y, optimizer, opt_state, model_shardings=None, opt_shardings=None):
    if model_shardings is not None:
        model = eqx.filter_shard(model, model_shardings)
        opt_state = eqx.filter_shard(opt_state, opt_shardings)
    value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
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


def validation_loss(
    model, data, key, batch_size, num_batches, sequence, data_sharding=None
):
    losses = []
    for _ in range(num_batches):
        key, bk = jax.random.split(key)
        x, y = data.get_batch(bk, batch_size, sequence, split="val")
        if data_sharding is not None:
            x = eqx.filter_shard(x, data_sharding)
            y = eqx.filter_shard(y, data_sharding)
        losses.append(loss(model, x, y))
    return np.mean(losses)


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str)
args = parser.parse_args()

cfg = load_config(config_path(args.config))

seed = cfg.get("seed", 0)
key = jax.random.PRNGKey(seed)
key, model_key = jax.random.split(key)

data = make_data(cfg)
config = model_config(cfg, data)
sequence = config.sequence

batch_size = cfg["training"]["batch_size"]
val_batch_size = cfg["validation"].get("batch_size", batch_size)

num_devices = jax.device_count()
assert (
    batch_size % num_devices == 0
), f"batch {batch_size} not divisible by {num_devices} gpus"
mesh = jax.make_mesh(
    (num_devices,), ("batch",), axis_types=(jax.sharding.AxisType.Auto,)
)
repl_sharding = NamedSharding(mesh, PartitionSpec())
data_sharding = NamedSharding(mesh, PartitionSpec("batch"))


def get_apply_shard(pattern):
    def apply_sharding(subtree):
        def leaf(x):
            if not isinstance(x, jax.ShapeDtypeStruct):
                return None
            if x.ndim < 2:
                return repl_sharding
            # A sparse linear's weight is (out, k) with k as small as 4, and its
            # gather indices are integers. Replicate anything the mesh cannot
            # divide rather than failing at trace time.
            axis = pattern.index("batch") if "batch" in pattern else None
            if axis is None or x.shape[axis] % num_devices != 0:
                return repl_sharding
            return NamedSharding(mesh, PartitionSpec(*pattern))

        return jax.tree.map(leaf, subtree)

    return apply_sharding


replicate = get_apply_shard(())
shard_axis_1 = get_apply_shard((None, "batch"))
shard_axis_0 = get_apply_shard(("batch", None))

init_outline = eqx.filter_eval_shape(lambda: create_model(config, model_key))

# Start from "replicate everything", then override only the tensors worth
# splitting. Mapping the whole tree up front also turns non-array leaves into
# None, which is what filter_shard expects — z1t keeps `MLP.act` (a plain
# function) as a pytree leaf, so a leaf-by-leaf build would miss it.
shard_outline = replicate(init_outline)
# logit head [vocab, embed], shard input
shard_outline = eqx.tree_at(
    lambda x: x.clf, shard_outline, shard_axis_1(init_outline.clf)
)
for counter in range(len(init_outline.blocks)):
    # qkv / out projections [*, embed], shard input
    shard_outline = eqx.tree_at(
        lambda x, c=counter: x.blocks[c].attn,
        shard_outline,
        shard_axis_1(init_outline.blocks[counter].attn),
    )
    # [4 * embed, embed], shard input
    shard_outline = eqx.tree_at(
        lambda x, c=counter: x.blocks[c].mlp.proj1,
        shard_outline,
        shard_axis_1(init_outline.blocks[counter].mlp.proj1),
    )
    # [embed, 4 * embed], shard output
    shard_outline = eqx.tree_at(
        lambda x, c=counter: x.blocks[c].mlp.proj2,
        shard_outline,
        shard_axis_0(init_outline.blocks[counter].mlp.proj2),
    )

model = eqx.filter_shard(create_model(config, model_key), shard_outline)
num_params = sum(
    x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))
)
num_active = active_params(model)

steps = cfg["training"]["steps"]
opt_cfg = cfg["optimizer"]
lr = opt_cfg.get("lr", 3e-4)
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=lr,
    warmup_steps=opt_cfg.get("warmup", 100),
    decay_steps=steps,
    end_value=lr * 0.1,
)
optimizer = optax.chain(
    optax.clip_by_global_norm(opt_cfg.get("grad_clip", 1.0)),
    optax.adamw(
        schedule,
        b1=opt_cfg.get("b1", 0.9),
        b2=opt_cfg.get("b2", 0.95),
        weight_decay=opt_cfg.get("weight_decay", 0.1),
    ),
)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
param_shardings = jax.tree.map(
    lambda p: p.sharding, eqx.filter(model, eqx.is_inexact_array)
)
opt_shardings = eqx.tree_at(
    lambda o: (o[1][0].mu, o[1][0].nu),
    jax.tree.map(lambda _: repl_sharding, opt_state),
    (param_shardings, param_shardings),
)
opt_state = eqx.filter_shard(opt_state, opt_shardings)

print("num params", num_params, "| active", num_active)

log_every = cfg["training"].get("log_every", 1)
val_log = cfg["training"].get("val_log", 100)
val_batches = cfg["validation"].get("batches", 16)
name = run_name(cfg, config, data.name)

wandb_cfg = cfg["wandb"]
run = wandb.init(
    project=wandb_cfg.get("project", "z1t"),
    entity=wandb_cfg.get("entity"),
    group=wandb_cfg.get("group"),
    mode=wandb_cfg.get("mode", "online"),
    name=name,
    config={
        **cfg,
        "model": asdict(config),
        "num_active_params": num_active,
        "num_params": num_params,
        "batch_size": batch_size,
    },
)

pbar = tqdm(range(0, steps + 1), total=steps, initial=0, desc=name)
for i in pbar:
    key, bk = jax.random.split(key)
    x, y = data.get_batch(bk, batch_size, sequence, split="train")
    x = eqx.filter_shard(x, data_sharding)
    y = eqx.filter_shard(y, data_sharding)
    model, train_loss, opt_state = step(
        model,
        x,
        y,
        optimizer,
        opt_state,
        model_shardings=shard_outline,
        opt_shardings=opt_shardings,
    )

    if i % log_every == 0:
        train_loss = float(train_loss)
        log = {
            "step": i,
            "train/loss": train_loss,
            "train/bpb": train_loss / np.log(2),
            "lr": float(schedule(i)),
            "tokens_seen": i * batch_size * sequence,
            "examples_seen": i * batch_size,
        }
        if i % val_log == 0:
            key, ek = jax.random.split(key)
            val_loss = validation_loss(
                model,
                data,
                ek,
                batch_size=val_batch_size,
                num_batches=val_batches,
                sequence=sequence,
                data_sharding=data_sharding,
            )
            log["val/loss"] = val_loss
            log["val/bpb"] = val_loss / np.log(2)
        wandb.log(log, step=i)

run.finish()
