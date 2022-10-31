from typing import Dict, Callable
import os

import json
from tqdm import trange

from absl import app
from absl import flags
from absl import logging

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

import flax.linen as nn
import optax

from transformers import AutoConfig
from sklearn.metrics import classification_report

from sepll.data.transform import add_unlabeled_randomly
from sepll.data.load_knodle import get_weak_tokenized_data
from sepll.data.iterators import Z_train_data_collator, Z_eval_data_collator

from sepll.training.state import TrainState
from sepll.training.optim import clipped_adamw
from sepll.training.learning_rate import decay_mask_fn, create_learning_rate_fn_by_steps

from sepll.training.regularization import l2_lf_matrix

from sepll.model import SepLLModel

"""
Parts are taken from :
- https://github.com/huggingface/transformers/blob/master/examples/flax/text-classification/run_flax_glue.py
- Flax Examples MNIST
"""

FLAGS = flags.FLAGS

# data flags
flags.DEFINE_string(
    "data_dir", default="",
    help="Data directory. Here, data is loaded for computation."
)

flags.DEFINE_string(
    "work_dir", default="",
    help="Working directory. Here, data is saved."
)

flags.DEFINE_bool(
    "debug", default=False,
    help="Whether this should only be debugging, meaning 1 iteration, 1 epoch."
)

flags.DEFINE_string(
    "objective", default="",
    help="Value in the output dictionary which is used to steer the hyperparameter optimization."
)

flags.DEFINE_string(
    "dataset", default="trec",
    help="Data set name to train on."
)

# training flags

flags.DEFINE_integer(
    "max_length", default=128,
    help="The maximum token length of an input."
)

flags.DEFINE_float(
    'learning_rate', default=1e-5,
    help='The learning rate for the Adam optimizer.'
)

flags.DEFINE_float(
    'warmup_steps', default=0,
    help="Number of steps for the warmup in the lr scheduler."
)

flags.DEFINE_float(
    'weight_decay', default=0.001,
    help='The learning rate for the Adam optimizer.'
)

flags.DEFINE_float(
    "momentum", default=0.9,
    help="the `decay` rate used by the momentum term"
)

flags.DEFINE_integer(
    'batch_size', default=64,
    help='Batch size for training.'
)

flags.DEFINE_integer(
    'num_steps', default=30,
    help='Number of training epochs.'
)

flags.DEFINE_float(
    'min_delta', default=0.2,
    help="Define a minimum amount of change for early stopping."
)

flags.DEFINE_float(
    'patience', default=5,
    help="Define a minimum amount of patience epochs for early stopping."
)

# model flags

flags.DEFINE_string(
    "z_noise_type", default="class",
    help="The type of noise added to the Z matrix."
)

flags.DEFINE_float(
    "z_noise", default=0.0,
    help="The amount of noisy labels which are included."
)

flags.DEFINE_string(
    "add_unlabelled", default="even",
    help="Which method to use to add unlabelled data to the training process."
)

flags.DEFINE_float(
    "l2_lf_regularization", default=0.0,
    help="L2 regularization applied to the LF part of the splitted code."
)


def compute_metrics(logits, labels):
    """Computes metrics and returns them."""
    logits = jnp.round(logits)
    metrics = {
        "accuracy": jnp.mean(logits == labels)
    }
    return metrics


@jax.jit
def train_step(state, batch, rng):
    labels = batch.pop("z")

    def loss_fn(params):
        y_probs, lambda_logits, pred_logits = state.apply_fn(**batch, params=params, dropout_rng=rng, train=True)
        loss = state.loss_fn(logits=pred_logits, labels=labels).mean()

        lf_l2_reg = l2_lf_matrix(params=params)
        loss += FLAGS.l2_lf_regularization * lf_l2_reg

        return loss, (lambda_logits, lf_l2_reg)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (lambda_logits, lf_l2_reg)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    metrics = compute_metrics(state.logits_fn(lambda_logits), labels)
    metrics["loss"] = loss
    metrics["lf_l2_regularization"] = lf_l2_reg
    metrics["lf_l2_regularization_alpha"] = FLAGS.l2_lf_regularization

    return state, metrics


@jax.jit
def eval_step(state, batch):
    y_probs, lambda_logits, pred_logits = state.apply_fn(**batch, params=state.params, train=False)
    return y_probs, lambda_logits, pred_logits


def test_model(state, embeddings, labels, Z):
    y_pred, y_truth = [], []
    Z_pred, Z_latent_pred, Z_truth = [], [], []

    # we also add new truth vectors, as a debuggin run will only yield the first batch
    for batch in Z_eval_data_collator(X=embeddings, y=labels, Z=Z, batch_size=FLAGS.batch_size):
        l, z = batch.pop("labels"), batch.pop("z")
        y_truth.append(l)
        Z_truth.append(z)

        y_probs, lambda_logits, pred_logits = eval_step(state, batch)

        y_pred.append(np.argmax(y_probs, axis=-1))
        Z_latent_pred.append(nn.sigmoid(lambda_logits))
        Z_pred.append(pred_logits)

    # construct arrays
    y_pred = np.concatenate(y_pred)
    Z_pred = np.concatenate(Z_pred)
    Z_latent_pred = np.concatenate(Z_latent_pred)

    y_truth = np.concatenate(y_truth)
    Z_truth = np.concatenate(Z_truth)

    # compute evaluations
    y_report = classification_report(y_truth, y_pred=y_pred, output_dict=True, zero_division=0)

    lambda_report = classification_report(Z_truth, y_pred=(Z_pred >= 0.5), output_dict=True, zero_division=0)
    lambda_report["accuracy"] = (pd.Series(Z_truth.flatten()) == pd.Series((Z_pred >= 0.5).flatten())).mean()

    lambda_latent_report = classification_report(
        Z_truth, y_pred=(Z_latent_pred >= 0.5), output_dict=True, zero_division=0
    )
    lambda_latent_report["accuracy"] = (
            pd.Series(Z_truth.flatten()) == pd.Series((Z_latent_pred >= 0.5).flatten())
    ).mean()

    report = {
        "y_metrics": y_report,
        "lambda_metrics": lambda_report,
        "lambda_latent_metrics": lambda_latent_report
    }
    return report


def create_roberta_train_state(T):
    """Creates initial `TrainState`."""

    config = AutoConfig.from_pretrained("roberta-base")
    model = SepLLModel.from_pretrained(
        "roberta-base",
        config=config,
        T=T
    )

    learning_rate_fn = create_learning_rate_fn_by_steps(
        num_train_steps=FLAGS.num_steps, num_warmup_steps=FLAGS.warmup_steps, learning_rate=FLAGS.learning_rate
    )

    tx = clipped_adamw(
        learning_rate=learning_rate_fn, b1=0.9, b2=0.999, eps=1e-6, weight_decay=FLAGS.weight_decay, mask=decay_mask_fn,
        clipping_threshold=1.0
    )

    return model, TrainState.create(
        apply_fn=model.__call__, params=model.params, tx=tx,
        logits_fn=nn.sigmoid, loss_fn=optax.softmax_cross_entropy
    )


def save_metrics(report: Dict, file_name: str):
    if FLAGS.objective in ["acc", "accuracy"]:
        report[FLAGS.objective] = report.get("y_metrics").get("accuracy")
    elif FLAGS.objective in ["f1", "f1-score"]:
        report[FLAGS.objective] = report.get("y_metrics").get("1").get("f1-score")

    with open(os.path.join(FLAGS.work_dir, f"{file_name}.json"), "w") as f:
        json.dump(report, f)
    return report


def main(_):  # -> train_state.TrainState:
    """Execute model training and evaluation loop.
    Args:
      config: Hyperparameter configuration for training and evaluation.
      work_dir: Directory where the tensorboard summaries are written to.
    Returns:
      The train state (which includes the `.params`).
    """

    # instantiate logging
    logging.debug("Set up helpers.")

    # instantiate model; termination through early stopping
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    logging.debug("Load data.")
    (
        X_train, y_train, X_unlabelled, T, (Z_labelled, Z_unlabelled),
        X_dev, y_dev, Z_dev,
        X_test, y_test, Z_test
    ) = get_weak_tokenized_data(data_dir=os.path.join(FLAGS.data_dir, FLAGS.dataset), max_length=FLAGS.max_length)

    if FLAGS.add_unlabelled == "even":
        X_train, Z_train = add_unlabeled_randomly(X_train, Z_labelled, X_unlabelled)
    else:
        Z_train = Z_labelled

    T = jnp.asarray(T.astype(np.bool))

    logging.debug("Load Model.")
    model, state = create_roberta_train_state(T=T)

    logging.info("Start Training.")

    best_step = -1
    best_step_dev_metric = -1

    step = 0
    history = {}
    last_step_log = {}
    ## Code is made similar to WRENCH:
    with trange(FLAGS.num_steps, desc=f"[FINETUNE] Classifier", unit="steps", disable=False,
                ncols=350, position=0, leave=True) as pbar:
        while step < FLAGS.num_steps:

            rng, perm_rng = jax.random.split(rng)

            train_metrics = []

            for batch in Z_train_data_collator(
                    X_train, Z_train, T=T, batch_size=FLAGS.batch_size, rng=perm_rng
            ):

                rng, step_rng = jax.random.split(rng)
                state, metrics = train_step(state, batch, step_rng)
                train_metrics.append(metrics)

                step += 1
                if step >= FLAGS.num_steps:
                    break

                if X_dev is not None and step % 10 == 0:
                    log_train_metrics = {k: np.mean([dic[k] for dic in train_metrics]) for k in train_metrics[0]}

                    with open(os.path.join(FLAGS.work_dir, f"step_{step}_train_metrics.json"), "w") as f:
                        json.dump({k: float(v) for k, v in log_train_metrics.items()}, f)

                    dev_report = test_model(state, X_dev, y_dev, Z_dev)
                    dev_report = save_metrics(dev_report, f"epoch_{step}_dev_metrics")

                    test_report = test_model(
                        state, X_test, y_test, Z_test
                    )
                    test_report = save_metrics(test_report, f"epoch_{step}_test_metrics")

                    if dev_report.get(FLAGS.objective) > best_step_dev_metric:
                        best_step = step
                        best_step_dev_metric = dev_report.get(FLAGS.objective)

                        model_save_path = os.path.join(FLAGS.work_dir, "models")
                        os.makedirs(model_save_path, exist_ok=True)
                        model.params = state.params
                        model.save_pretrained(model_save_path)
                        with open(os.path.join(FLAGS.work_dir, "stats.json"), "w") as f:
                            json.dump({
                                "step": step,
                                f"dev_{FLAGS.objective}": dev_report.get(FLAGS.objective),
                                f"test_{FLAGS.objective}": test_report.get(FLAGS.objective)
                            }, f)

                        save_metrics(test_report, "test_metrics")

                    history[step] = {
                        'loss': log_train_metrics.get("loss"),
                        f'val_{FLAGS.objective}': dev_report.get(FLAGS.objective),
                        f'best_val_{FLAGS.objective} ': best_step_dev_metric,
                        'best_step': best_step,
                    }
                    last_step_log.update(history[step])

                last_step_log['loss'] = train_metrics[-1].get("loss")
                pbar.update()
                pbar.set_postfix(ordered_dict=last_step_log)

    if best_step == -1:
        test_report = test_model(
            state, X_test, y_test, Z_test
        )

        save_metrics(test_report, "test_metrics")


if __name__ == '__main__':
    app.run(main)
