import argparse
import os
from dataclasses import asdict

import equinox as eqx
import jax
import numpy as np
import optax
import wandb
import yaml
from jax.sharding import NamedSharding, PartitionSpec
from tqdm import tqdm

from st import active_params, Config, flops, Transformer
from st.dataset import TextDataLoader
from st.tokenizer import ByteTokenizer
from st.train import loss, step


def validation_loss(
    model, dataloader, batch_size, num_batches, tokenizer, data_sharding=None
):
    losses = []

    for _ in range(num_batches):
        batch = dataloader.sample_val_batch(batch_size)
        if data_sharding is not None:
            batch = eqx.filter_shard(batch, data_sharding)
        losses.append(loss(model, batch, tokenizer))

    return np.mean(losses)


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str)
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

seed = cfg["seed"]
key = jax.random.key(seed)

tokenizer = ByteTokenizer()
dataloader = TextDataLoader(
    os.path.join(cfg["data_path"], "train.npy"),
    os.path.join(cfg["data_path"], "validation.npy"),
    cfg["model"]["sequence"],
)

config = Config(**cfg["model"])
batch_size = cfg["training"]["batch_size"]
val_batch_size = cfg["validation"]["batch_size"]

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
            return NamedSharding(mesh, PartitionSpec(*pattern))

        return jax.tree.map(leaf, subtree)

    return apply_sharding


replicate = get_apply_shard(())
shard_axis_1 = get_apply_shard((None, "batch"))
shard_axis_0 = get_apply_shard(("batch", None))


init_outline = eqx.filter_eval_shape(lambda: Transformer(config, key))
# embeddings
shard_outline = eqx.tree_at(lambda x: x.wte, init_outline, replicate(init_outline.wte))
shard_outline = eqx.tree_at(
    lambda x: x.wpe, shard_outline, replicate(shard_outline.wpe)
)
# norm
shard_outline = eqx.tree_at(
    lambda x: x.norm, shard_outline, replicate(shard_outline.norm)
)
# output linear [vocab, embed]
shard_outline = eqx.tree_at(
    lambda x: x.output_linear, shard_outline, shard_axis_1(shard_outline.output_linear)
)
counter = 0
for _ in range(len(shard_outline.h)):
    # QKV = [embed, embed], shard input
    shard_outline = eqx.tree_at(
        lambda x: x.h[counter].attn,
        shard_outline,
        shard_axis_1(shard_outline.h[counter].attn),
    )
    shard_outline = eqx.tree_at(
        lambda x: x.h[counter].norm1,
        shard_outline,
        replicate(shard_outline.h[counter].norm1),
    )
    shard_outline = eqx.tree_at(
        lambda x: x.h[counter].norm2,
        shard_outline,
        replicate(shard_outline.h[counter].norm2),
    )
    # [4 * embed, embed], shard input
    shard_outline = eqx.tree_at(
        lambda x: x.h[counter].ffn.lin1,
        shard_outline,
        shard_axis_1(shard_outline.h[counter].ffn.lin1),
    )
    # [embed, 4 * embed], shard output
    shard_outline = eqx.tree_at(
        lambda x: x.h[counter].ffn.lin2,
        shard_outline,
        shard_axis_0(shard_outline.h[counter].ffn.lin2),
    )
    counter += 1


shard_outline = eqx.tree_at(
    lambda x: x.seq_len, shard_outline, None, is_leaf=lambda x: x is None
)
model = eqx.filter_shard(Transformer(config, key), shard_outline)
num_params = sum(
    x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))
)
num_active = active_params(model)
flops_per = flops(config)


steps = cfg["training"]["steps"]
schedule = optax.cosine_decay_schedule(
    init_value=cfg["optimizer"]["lr"],
    decay_steps=steps,
    alpha=0.1,
)

optimizer = optax.adamw(
    learning_rate=schedule,
    weight_decay=cfg["optimizer"]["weight_decay"],
)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
param_shardings = jax.tree.map(
    lambda p: p.sharding, eqx.filter(model, eqx.is_inexact_array)
)
opt_shardings = eqx.tree_at(
    lambda o: (o[0].mu, o[0].nu),
    jax.tree.map(lambda _: repl_sharding, opt_state),
    (param_shardings, param_shardings),
)
opt_state = eqx.filter_shard(opt_state, opt_shardings)
total_flops = 0

print("num params", num_params, "| active", num_active)
print("flops/", flops_per)

val_log = cfg["training"]["val_log"]
val_batches = cfg["validation"]["batches"]


parts = [
    cfg["name"],
    f"L{config.n_layers}",
    f"D{config.n_embed}",
    f"H{config.n_head}",
]
if config.attn_window is not None:
    parts.append(f"AW{config.attn_window}")
if config.linear_fan_in is not None:
    parts.append(f"LF{config.linear_fan_in}")
run_name = "-".join(parts)

_wandb_kw = dict(
    project=cfg["wandb"]["project"],
    entity=cfg["wandb"].get("entity"),
    group=cfg["wandb"].get("group"),
    name=run_name,
    config={
        **cfg,
        "model": asdict(config),
        "num_active_params": num_active,
        "flops_per_example": flops_per,
        "train_tokens": len(dataloader.train),
        "num_params": num_params,
        "batch_size": batch_size,
    },
)

run = wandb.init(**_wandb_kw)

pbar = tqdm(range(0, steps + 1), total=steps, initial=0, desc=run_name)
for i in pbar:
    batch = dataloader.sample_train_batch(batch_size)
    batch = eqx.filter_shard(batch, data_sharding)
    model, train_loss, opt_state = step(
        model,
        batch,
        optimizer,
        opt_state,
        tokenizer,
        model_shardings=shard_outline,
        opt_shardings=opt_shardings,
    )
    total_flops += flops_per * batch_size

    log = {
        "step": i,
        "train/loss": float(train_loss),
        "total_flops": total_flops,
        "tokens_seen": i * batch_size * config.sequence,
        "examples_seen": i * batch_size,
        "train/bpb": train_loss / np.log(2),
    }

    if i % val_log == 0 or i == 1:
        val_loss = validation_loss(
            model,
            dataloader,
            batch_size=val_batch_size,
            num_batches=val_batches,
            tokenizer=tokenizer,
            data_sharding=data_sharding,
        )
        log["val/loss"] = val_loss
        log["val/bpb"] = val_loss / np.log(2)

    wandb.log(log, step=i)


run.finish()
