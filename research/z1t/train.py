#!/usr/bin/env python
"""Train a z1t AFT transformer. The one entrypoint into this package.

    python -m z1t.train --config configs/z1t_addition_aft_conv.yaml

Same shape as `experiments/sparse_gpt_scaling.py`: a run is one nested yaml, `--config`
points at it, and `model.vocab` / `model.sequence` are filled in from the dataset
rather than the yaml, so a config cannot disagree with the data it trains on.
There are no architecture flags — if a knob changes the model, it lives in the
yaml.

One eqx.filter_jit'd step, compiled once (fixed B, T) with buffers donated. The
Python loop only syncs on the device when it logs, which is why train loss goes
to wandb every `training.log_every` steps rather than every step.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict

import equinox as eqx
import jax
import numpy as np
import optax
import wandb
import yaml
from jax.sharding import PartitionSpec as P
from tqdm import tqdm

from z1t.components import AFT_KINDS, Config, active_params
from z1t.dataset import (
    make_addition,
    make_openwebtext,
    make_shakespeare,
    make_wikitext,
)
from z1t.model import create_model

# Config paths are resolved relative to the repo root, so a run works from any
# working directory (repo root, a cluster scratch dir, a Modal container). The
# package lives at research/z1t, so the root is two levels up, not one.
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))

# Top-level sections a run yaml may define. Anything else is a typo, not a
# silent no-op; unknown keys *inside* `model:` are caught by `Config(**block)`.
SECTIONS = {
    "seed",
    "name",
    "run_name",
    "out_dir",
    "data",
    "optimizer",
    "training",
    "validation",
    "wandb",
    "model",
}


# === config ===


def config_path(path):
    """Resolve a config path: as given, else relative to the repo root.

    So `--config configs/z1t_tiny.yaml` works from the repo root AND from a
    cluster scratch dir or a container where cwd is elsewhere. Absolute paths
    and paths that exist as given are returned untouched.
    """
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(REPO_ROOT, path)
    return candidate if os.path.exists(candidate) else path


def load_config(path):
    """Read a run yaml and check its top-level shape. The only disk read."""
    with open(config_path(path)) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    bad = set(cfg) - SECTIONS
    if bad:
        raise ValueError(f"unknown keys in {path}: {sorted(bad)}")
    for section in ("data", "optimizer", "training", "validation", "wandb", "model"):
        cfg.setdefault(section, {})
    return cfg


def make_data(cfg):
    """Build the dataset named by `data.dataset_name`."""
    data_cfg = cfg["data"]
    name = data_cfg.get("dataset_name", "shakespeare")
    data_path = data_cfg.get("data_path", "./data")
    n_tokens = data_cfg.get("n_tokens", 0) or None
    if name == "shakespeare":
        return make_shakespeare(data_path, data_cfg.get("val_frac", 0.1))
    if name == "wikitext":
        return make_wikitext(
            data_path, data_cfg.get("val_frac", 0.1), n_tokens=n_tokens
        )
    if name == "openwebtext":
        return make_openwebtext(data_path, n_tokens=n_tokens)
    if name == "addition":
        return make_addition(
            max_n_digits=data_cfg.get("max_n_digits", 3),
            n_tokens=n_tokens,
            pool_seed=data_cfg.get("data_seed", 0),
        )
    raise ValueError(f"unknown dataset_name: {name!r}")


def model_config(cfg, data) -> Config:
    """Build the model `Config` from the yaml's `model:` block plus the dataset.

    `vocab` always comes from the dataset. So does `sequence` for a fixed-length
    corpus (addition is exactly `sample_len - 1` tokens of input); for the
    free-length text corpora the yaml has to say. A yaml that contradicts a
    fixed-length dataset is an error rather than a silently mis-shaped model.
    """
    block = dict(cfg["model"])
    block["vocab"] = data.vocab_size

    forced = None if data.sample_len is None else data.sample_len - 1
    if forced is not None:
        if block.get("sequence", forced) != forced:
            raise ValueError(
                f"model.sequence={block['sequence']} contradicts dataset "
                f"{data.name}, which is fixed at {forced}"
            )
        block["sequence"] = forced
    elif "sequence" not in block:
        raise ValueError(f"model.sequence is required for dataset {data.name}")

    config = Config(**block)  # unknown model keys raise here
    if config.aft_kind not in AFT_KINDS:
        raise ValueError(
            f"unknown aft_kind: {config.aft_kind!r} (expected one of {AFT_KINDS})"
        )
    return config


def run_name(cfg, config: Config, data_name: str) -> str:
    """One self-describing run name — wandb name and checkpoint stem.

    Same assembly as `experiments/sparse_gpt_scaling.py`: a `name:` stem, the always-on
    shape, then one tag per knob that is actually active, joined with `-`. An
    explicit `run_name:` in the yaml wins outright.
    """
    if cfg.get("run_name"):
        return cfg["run_name"]

    parts = [cfg.get("name", "z1t"), data_name]
    parts += [f"L{config.n_layers}", f"D{config.n_embed}", f"T{config.sequence}"]

    attn = f"AFT-{config.aft_kind}"
    if config.aft_kind == "conv":
        attn += f"H{config.aft_heads}K{config.aft_ksize}"
    parts.append(attn)

    parts.append(f"A{config.dyt_alpha}")
    for site in ("attn", "mlp"):
        fan = config.fan_in(site)
        if fan is not None:
            parts.append(f"{site.upper()}F{fan}")
    if config.tanh_linear:
        parts.append("TL")
    if config.tanh_mlp:
        parts.append("TM")
    if config.remat:
        parts.append("RM")

    return "-".join(parts)


# === loss ===


@eqx.filter_value_and_grad
def loss_and_grad(model, x, y):
    # x, y: (B, T). The model maps (T,) -> (T, V); vmap over the batch.
    logits = jax.vmap(model)(x)                                   # (B, T, V)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return losses.mean()


# === train / eval steps ===
# Both are jitted. They are pure functions of their array arguments, so with
# fixed (B, T) each compiles once and is reused for the whole run.


def make_steps(optim):
    @eqx.filter_jit(donate="all")            # reuse model + opt_state buffers in place
    def train_step(model, opt_state, x, y):
        loss, grads = loss_and_grad(model, x, y)
        params = eqx.filter(model, eqx.is_inexact_array)
        updates, opt_state = optim.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    @eqx.filter_jit
    def eval_step(model, x, y):
        logits = jax.vmap(model)(x)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        return losses.mean()

    return train_step, eval_step


def validation_loss(
    eval_step, model, data, key, batch_size, batches, sequence, split="val"
):
    """Mean cross-entropy over `batches` held-out batches."""
    total = 0.0
    for _ in range(batches):
        key, bk = jax.random.split(key)
        x, y = data.get_batch(bk, batch_size, sequence, split=split)
        total += eval_step(model, x, y).item()      # sync, but only at eval time
    return total / batches


def main():
    p = argparse.ArgumentParser(description="train a z1t AFT transformer")
    p.add_argument(
        "--config", type=str, required=True, help="run yaml; see configs/"
    )
    p.add_argument(
        "--print-model",
        action="store_true",
        help="print the model pytree and exit — no data, no device loop",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    train_cfg, val_cfg, opt_cfg = cfg["training"], cfg["validation"], cfg["optimizer"]

    seed = cfg.get("seed", 0)
    steps = train_cfg.get("steps", 5000)
    batch_size = train_cfg.get("batch_size", 64)
    log_every = train_cfg.get("log_every", 50)
    val_log = train_cfg.get("val_log", 250)
    val_batches = val_cfg.get("batches", 16)
    val_batch_size = val_cfg.get("batch_size", batch_size)
    out_dir = cfg.get("out_dir", ".")
    ddp = train_cfg.get("ddp", False)
    lr = opt_cfg.get("lr", 1e-3)

    print(f"devices: {jax.devices()}")

    if ddp:
        assert len(jax.devices()) > 1, "DDP assumes multiple devices"
        mesh = jax.make_mesh((len(jax.devices()),), axis_names=("B",))
        dev_batch = jax.NamedSharding(mesh, P("B", None))          # (B, T)
        dev_copy = jax.NamedSharding(mesh, P())

    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)

    data = make_data(cfg)
    config = model_config(cfg, data)
    sequence = config.sequence
    print(f"dataset: {data.name}  vocab={data.vocab_size}  sequence={sequence}")

    model = create_model(config, model_key)
    num_params = sum(
        x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))
    )
    num_active = active_params(model)
    name = run_name(cfg, config, data.name)
    print(f"num params {num_params} | active {num_active}")
    print(f"run: {name}")

    if args.print_model:
        print(eqx.tree_pformat(model))
        return

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
            "num_params": num_params,
            "num_active_params": num_active,
            "batch_size": batch_size,
        },
    )

    # optimizer
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=opt_cfg.get("warmup", 100),
        decay_steps=steps,
        end_value=lr * 0.1,
    )
    optim = optax.chain(
        optax.clip_by_global_norm(opt_cfg.get("grad_clip", 1.0)),
        optax.adamw(
            schedule,
            b1=opt_cfg.get("b1", 0.9),
            b2=opt_cfg.get("b2", 0.95),
            weight_decay=opt_cfg.get("weight_decay", 0.1),
        ),
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    if ddp:
        model = jax.device_put(model, dev_copy)
        opt_state = jax.device_put(opt_state, dev_copy)

    train_step, eval_step = make_steps(optim)

    # train loop
    tok_per_step = batch_size * sequence
    t0 = time.perf_counter()
    val = float("nan")           # last val loss, shown until refreshed
    tr_eval = float("nan")       # last train-split eval loss (held-out CE on train)

    pbar = tqdm(range(1, steps + 1), dynamic_ncols=True, desc=name)
    for step in pbar:
        key, bk = jax.random.split(key)
        x, y = data.get_batch(bk, batch_size, sequence, split="train")

        if ddp:
            x = jax.device_put(x, dev_batch)
            y = jax.device_put(y, dev_batch)

        model, opt_state, loss = train_step(model, opt_state, x, y)

        if step % log_every == 0:
            train_loss = loss.item()            # the one sync point in the hot loop
            dt = time.perf_counter() - t0
            tps = tok_per_step * log_every / dt
            cur_lr = float(schedule(step))
            pbar.set_postfix(
                loss=f"{train_loss:.4f}",
                val=f"{val:.4f}",
                gap=f"{val - tr_eval:+.3f}",
                lr=f"{cur_lr:.1e}",
                tps=f"{tps/1e3:.0f}k",
            )
            wandb.log(
                {
                    "train/loss": train_loss,
                    "train/bpb": train_loss / np.log(2),
                    "lr": cur_lr,
                    "tok_per_s": tps,
                    "tokens_seen": step * tok_per_step,
                    "examples_seen": step * batch_size,
                },
                step=step,
            )
            t0 = time.perf_counter()

        if step % val_log == 0:
            key, ek = jax.random.split(key)
            val = validation_loss(
                eval_step, model, data, ek, val_batch_size, val_batches, sequence,
                split="val",
            )
            key, ek = jax.random.split(key)
            tr_eval = validation_loss(
                eval_step, model, data, ek, val_batch_size, val_batches, sequence,
                split="train",
            )
            wandb.log(
                {
                    "val/loss": val,
                    "val/bpb": val / np.log(2),
                    "val/train_split_loss": tr_eval,
                    "val/gen_gap": val - tr_eval,
                },
                step=step,
            )
            t0 = time.perf_counter()

    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, f"{name}.eqx")
    eqx.tree_serialise_leaves(ckpt, model)
    print(f"checkpoint: {ckpt}")
    print(
        f"final  step={steps}  train_loss={loss.item():.4f}  "
        f"val_loss={val:.4f}  train_eval_loss={tr_eval:.4f}  "
        f"gen_gap={val - tr_eval:+.4f}"
    )

    run.finish()
    return model


if __name__ == "__main__":
    main()
