"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
def numerical_gradient(f, x, eps=1e-5):
    # TODO: Estimate the gradient of scalar f w.r.t. array x via central finite differences
    grad = np.zeros_like(x, dtype=float)
    if x.size == 0:
        return grad

    x_eval = np.array(x, dtype=float, copy=True)

    it = np.nditer(x_eval, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig_val = x_eval[idx]

        x_eval[idx] = orig_val + eps
        fx_plus = float(f(x_eval))

        x_eval[idx] = orig_val - eps
        fx_minus = float(f(x_eval))
        
        # Central difference formula
        grad[idx] = (fx_plus - fx_minus) / (2.0 * eps)

        # Restore perturbed element
        x_eval[idx] = orig_val
        it.iternext()

    return grad

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    # TODO: Return max relative error between analytic and numeric gradients.
    if analytic_grad.size == 0:
        return 0.0

    diff = np.abs(analytic_grad - numeric_grad)
    denom = np.maximum(
        np.maximum(np.abs(analytic_grad), np.abs(numeric_grad)), tol
    )
    return float(np.max(diff / denom))

# Step 3 - make_dense
def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer.

    Inputs:
      in_dim: int, input feature size
      out_dim: int, output feature size
      weight_init_fn: callable(in_dim, out_dim) -> (W, b)

    Returns layer dict with keys:
      params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
      forward(x) -> (y, cache) with y shape (batch, out_dim)
      backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
        Analytic dx/dW/db must match numerical_gradient via gradient_check.
    """
    # TODO: your approach here
    W, b = weight_init_fn(in_dim, out_dim)
    params = {"W": W, "b": b}

    def forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        W_curr = params["W"]
        b_curr = params["b"]

        y = x @ W_curr + b_curr
        cache = x
        return y, cache

    def backward(
        dout: np.ndarray, cache: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        x = cache
        W_curr = params["W"]

        dx = dout @ W_curr.T

        dW = x.T @ dout

        db = np.sum(dout, axis=0)

        grads = {"W": dW, "b": db}

        return dx, grads

    return {"params": params, "forward": forward, "backward": backward}

# Step 4 - make_activation
from typing import Any, Callable
import numpy as np

class ReLUStrategy:

  @staticmethod
  def forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.maximum(0.0, x)
    cache = x  
    return y, cache

  @staticmethod
  def backward(dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
    x = cache
    return np.where(x > 0.0, dout, 0.0)


class SigmoidStrategy:

  @staticmethod
  def forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos_mask = x >= 0
    y = np.empty_like(x, dtype=float)
    y[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[~pos_mask])
    y[~pos_mask] = exp_x / (1.0 + exp_x)
    cache = y 
    return y, cache

  @staticmethod
  def backward(dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
    y = cache
    return dout * y * (1.0 - y)


class TanhStrategy:
  @staticmethod
  def forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.tanh(x)
    cache = y 
    return y, cache

  @staticmethod
  def backward(dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
    y = cache
    return dout * (1.0 - y**2)


ACTIVATION_REGISTRY = {
    "relu": ReLUStrategy,
    "sigmoid": SigmoidStrategy,
    "tanh": TanhStrategy,
}

def make_activation(kind: str = "relu") -> dict[str, Any]:
  """Create a genuinely nonlinear elementwise activation layer.

  Args:
      kind: str nonlinearity name ('relu', 'sigmoid', 'tanh'). Default 'relu'.

  Returns:
      Layer dict with 'params', 'forward', and 'backward'.
  """
  strategy_cls = ACTIVATION_REGISTRY.get(kind.lower())
  if strategy_cls is None:
    valid_keys = list(ACTIVATION_REGISTRY.keys())
    raise ValueError(
        f"Unsupported activation kind: '{kind}'. Available: {valid_keys}"
    )

  params: dict[str, np.ndarray] = {}

  def forward(x: np.ndarray) -> tuple[np.ndarray, Any]:
    y, cache = strategy_cls.forward(x)
    return y, cache

  def backward(
      dout: np.ndarray, cache: Any
  ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    dx = strategy_cls.backward(dout, cache)
    return dx, {}

  return {"params": params, "forward": forward, "backward": backward}

# Step 5 - initialize_weights
def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer.

    Inputs:
      in_dim: int fan-in
      out_dim: int fan-out
      scheme: str initialization family (default 'he')

    Returns:
      W: np.ndarray shape (in_dim, out_dim), finite, symmetry-breaking,
         scale stable with depth (fan-in dependent)
      b: np.ndarray shape (out_dim,), near zero
    """
    # TODO: your approach here
    scheme_key = scheme.lower()

    if scheme_key in ("he", "kaiming"):
      std = np.sqrt(2.0 / in_dim)
    elif scheme_key in ("xavier", "glorot"):
      std = np.sqrt(2.0 / (in_dim + out_dim))
    elif scheme_key == "lecun":
      std = np.sqrt(1.0 / in_dim)
    else:
      raise ValueError(
          f"Unsupported initialization scheme: '{scheme}'. Choose from 'he',"
          " 'xavier', 'lecun'."
      )

    W = np.random.normal(0, std, size = (in_dim, out_dim)) 

    b = np.zeros(out_dim, dtype=float)

    return W, b

# Step 6 - make_loss
def make_loss(kind='cross_entropy'):
    """Return a classification loss_fn(logits, labels) -> (loss, d_logits).

    Inputs to loss_fn:
      logits: (batch, C) float array of raw class scores
      labels: (batch,) int array of class indices in [0, C)
    Outputs:
      loss: Python float, mean scalar loss over the batch (finite)
      d_logits: (batch, C) gradient of loss w.r.t. logits (finite)
    Must pass gradient_check, be minimized by confident correct predictions,
    and stay finite under saturated logits.
    """
    # TODO: your approach here
    if kind != "cross_entropy":
      raise ValueError(f"Unsupported loss kind: '{kind}'. Choose 'cross_entropy'.")

    def loss_fn(
        logits: np.ndarray, labels: np.ndarray
    ) -> tuple[float, np.ndarray]:
      """Compute mean cross-entropy loss and its analytic gradient w.r.t logits.

      Args:
          logits: Unnormalized class scores of shape (batch, C).
          labels: 1D array of integer ground-truth class indices in [0, C) of
            shape (batch,).

      Returns:
          loss: Python float, mean loss across the batch.
          d_logits: Gradient of loss w.r.t logits, shape (batch, C).
      """
      N, C = logits.shape
      if N == 0:
        return 0.0, np.zeros_like(logits, dtype=float)

      max_logits = np.max(logits, axis=1, keepdims=True)
      shifted = logits - max_logits
      exp_shifted = np.exp(shifted)
      sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)

      probs = exp_shifted / sum_exp

      # Sample loss: L_i = -log(p_{i, y_i}) = log(sum(exp(z - m))) - (z_{y_i} - m)
      batch_idx = np.arange(N)
      log_sum_exp = np.log(sum_exp).squeeze(axis=1)
      correct_shifted = shifted[batch_idx, labels]
      sample_losses = log_sum_exp - correct_shifted

      loss = float(np.mean(sample_losses))

      # Analytic gradient w.r.t logits: dL / dz_{i, c} = (p_{i, c} - 1[c == y_i]) / N
      d_logits = probs.copy()
      d_logits[batch_idx, labels] -= 1.0
      d_logits /= N

      return loss, d_logits

    return loss_fn

# Step 7 - make_sequential
def make_sequential(layers):
    """Compose protocol-honoring layers into one sequential model.

    Inputs:
      layers: list of layer dicts, each with
        forward(x) -> (y, cache),
        backward(dout, cache) -> (dx, grads_dict),
        params: dict of ndarrays (possibly empty).

    Returns a dict with:
      forward(x) -> (y, caches)
        y: final activation after applying every layer in order
        caches: opaque structure needed by backward
      backward(dout, caches) -> (dx, grads_list)
        dx: gradient w.r.t. the original input x
        grads_list: list of length len(layers); grads_list[i] is the
          grads_dict from layers[i] ({} for param-free layers)
      params: aggregated live view of all layer params, length len(layers),
        same order as layers (so in-place updates affect the model)
    """
    # TODO: your approach here
    params = [layer["params"] for layer in layers]

    def forward(x):
      caches = []
      out = x
      for layer in layers:
        out, cache = layer["forward"](out)
        caches.append(cache)
      return out, caches

    def backward(dout, caches):
      grads_list = [None] * len(layers)
      din = dout
      for i in reversed(range(len(layers))):
        din, grads = layers[i]["backward"](din, caches[i])
        grads_list[i] = grads
      return din, grads_list

    return {
      "forward": forward,
      "backward": backward,
      "params": params,
    }

# Step 8 - forward_backward
def forward_backward(model, loss_fn, x, y):
    """Run one full forward-backward sweep on a batch.

    Inputs:
      model: sequential dict with 'forward', 'backward', 'params'
             model['forward'](x) -> (logits, caches)
             model['backward'](d_logits, caches) -> (dx, param_grads)
      loss_fn: callable (logits, y) -> (loss, d_logits)
      x: np.ndarray (batch, in_dim)
      y: np.ndarray (batch,) integer labels

    Returns:
      loss: float, scalar batch loss
      param_grads: nested np.ndarrays matching model['params'] layout
                   (gradients of loss w.r.t. every parameter)
    """
    # TODO: your approach here
    logits, caches = model["forward"](x)

    loss, d_logits = loss_fn(logits, y)

    _, param_grads = model["backward"](d_logits, caches)

    return float(loss), param_grads

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

