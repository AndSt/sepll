from absl import flags
from tqdm import tqdm

import numpy as np

import jax
import jax.numpy as jnp
from jax.random import PRNGKey

FLAGS = flags.FLAGS


def refine_Z_by_class(Z, T):
    if FLAGS.z_noise == 0:
        return Z

    exist_classes = np.dot(Z, T) > 0
    sample_classes = np.dot(exist_classes, T.T)

    sample = np.random.binomial(n=1, p=FLAGS.z_noise, size=Z.shape)
    Z_new = Z + sample * sample_classes
    np.clip(Z_new.round(), a_min=0, a_max=1)
    return Z_new


def refine_Z_full(Z, T):
    if FLAGS.z_noise == 0:
        return Z

    Z_new = Z + np.random.uniform(0, FLAGS.z_noise, size=Z.shape)

    return Z_new


def Z_train_data_collator(X, Z, T, batch_size: int, rng: PRNGKey):
    """Returns shuffled batches of size `batch_size` from truncated `train dataset`, sharded over all local devices."""
    dataset_len = Z.shape[0]
    steps_per_epoch = dataset_len // batch_size
    perms = jax.random.permutation(rng, dataset_len)
    perms = perms[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    perms = perms.reshape((steps_per_epoch, batch_size))

    perms = perms[0: 1, :] if flags.FLAGS.debug else perms

    if FLAGS.z_noise_type == "class":
        Z = refine_Z_by_class(Z, T)
    elif FLAGS.z_noise_type == "full":
        Z = refine_Z_full(Z, T)

    # normalize Z
    Z = Z / jnp.sum(Z, axis=1, keepdims=True)

    for perm in tqdm(perms):
        batch = {
            "input_ids": X[0][perm, ...],
            "attention_mask": X[1][perm, ...],
            "z": Z[perm, ...]
        }
        yield batch


def Z_eval_data_collator(X, y, Z, batch_size: int):
    """Returns batches of size `batch_size` from `eval dataset`, sharded over all local devices."""

    num_its = 1 if flags.FLAGS.debug else y.shape[0] // batch_size

    for i in tqdm(range(1 + num_its)):
        batch = {
            "input_ids": X[0][i * batch_size: (i + 1) * batch_size],
            "attention_mask": X[1][i * batch_size: (i + 1) * batch_size],
            "labels": y[i * batch_size: (i + 1) * batch_size],
            "z": Z[i * batch_size: (i + 1) * batch_size]
        }
        yield batch
