from typing import Optional, Any, Union, Callable

from optax._src.alias import ScalarOrSchedule, _scale_by_learning_rate
from optax._src import base
from optax._src import combine
from optax._src import clipping
from optax._src import transform


def clipped_adamw(
        learning_rate: ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        mu_dtype: Optional[Any] = None,
        weight_decay: float = 1e-4,
        mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
        clipping_threshold: Optional[float] = 1.0,
) -> base.GradientTransformation:
    """See optax.alias for more details on adamw.
    We only added clipping to the chaining
    """
    return combine.chain(
        clipping.clip_by_global_norm(clipping_threshold),
        transform.scale_by_adam(
            b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
        transform.add_decayed_weights(weight_decay, mask),
        _scale_by_learning_rate(learning_rate),
    )
