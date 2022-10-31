from typing import Callable

from flax import struct
from flax.training import train_state


class TrainState(train_state.TrainState):
    """Train state with an Optax optimizer.
    The two functions below differ depending on whether the task is classification
    or regression.
    Args:
      logits_fn: Applied to last layer to obtain the logits.
      loss_fn: Function to compute the loss.
    """

    logits_fn: Callable = struct.field(pytree_node=False)
    loss_fn: Callable = struct.field(pytree_node=False)