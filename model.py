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

    W = np.random.randn(in_dim, out_dim) * std 

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

# Step 9 - make_optimizer
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Type, Union
import numpy as np

def traverse_params_grads(params: Any, grads: Any, callback: Callable) -> None:
  """Walks any arbitrary nested tree of params and grads in lockstep."""
  if isinstance(params, np.ndarray) and isinstance(grads, np.ndarray):
    if params.size > 0:
      callback(params, grads)
  elif isinstance(params, dict) and isinstance(grads, dict):
    for k, p_val in params.items():
      if k in grads:
        traverse_params_grads(p_val, grads[k], callback)
  elif isinstance(params, (list, tuple)) and isinstance(grads, (list, tuple)):
    for p_item, g_item in zip(params, grads):
      traverse_params_grads(p_item, g_item, callback)

class BaseOptimizer(ABC):

  def __init__(self, params: Any, lr: float = 1e-2):
    if lr <= 0.0:
      raise ValueError(f"Learning rate must be positive, got {lr}")
    self.params = params
    self.lr = float(lr)
    self.state: Dict[int, Dict[str, Any]] = {}
    self._init_state()

  def _init_state(self) -> None:
    """Collects and initializes state for each parameter ndarray."""

    def _register(p: np.ndarray, _):
      self.state[id(p)] = self._init_param_state(p)

    traverse_params_grads(self.params, self.params, _register)

  @abstractmethod
  def _init_param_state(self, p: np.ndarray) -> Dict[str, Any]:
    """Return initial buffer states for a single parameter tensor."""
    pass

  @abstractmethod
  def _update_param(
      self, p: np.ndarray, g: np.ndarray, state: Dict[str, Any]
  ) -> None:
    """Compute and apply in-place delta on p using its gradient and state."""
    pass

  def step(self, grads: Any) -> None:
    """Applies one optimization step mutating parameters in-place."""

    def _apply(p: np.ndarray, g: np.ndarray):
      p_id = id(p)
      if p_id not in self.state:
        self.state[p_id] = self._init_param_state(p)
      self._update_param(p, g, self.state[p_id])

    traverse_params_grads(self.params, grads, _apply)

  def as_contract_dict(self) -> Dict[str, Any]:
    """Returns standard dict interface required by the contract."""
    return {
        "step": self.step,
        "params": self.params,
        "lr": self.lr,
        "optimizer": self,
    }

class SGDOptimizer(BaseOptimizer):
  """Standard Stochastic Gradient Descent."""

  def _init_param_state(self, p: np.ndarray) -> Dict[str, Any]:
    return {}

  def _update_param(
      self, p: np.ndarray, g: np.ndarray, state: Dict[str, Any]
  ) -> None:
    delta = (self.lr * g).astype(p.dtype, copy=False)
    p[...] -= delta


class MomentumOptimizer(BaseOptimizer):
  """SGD with Polyak Momentum."""

  def __init__(self, params: Any, lr: float = 1e-2, momentum: float = 0.9):
    self.momentum = float(momentum)
    super().__init__(params, lr)

  def _init_param_state(self, p: np.ndarray) -> Dict[str, Any]:
    return {"velocity": np.zeros_like(p, dtype=float)}

  def _update_param(
      self, p: np.ndarray, g: np.ndarray, state: Dict[str, Any]
  ) -> None:
    v = state["velocity"]
    v[...] = self.momentum * v + g
    delta = (self.lr * v).astype(p.dtype, copy=False)
    p[...] -= delta


class RMSpropOptimizer(BaseOptimizer):
  """Root Mean Square Propagation (Hinton)."""

  def __init__(
      self,
      params: Any,
      lr: float = 1e-2,
      decay_rate: float = 0.99,
      eps: float = 1e-8,
  ):
    self.decay_rate = float(decay_rate)
    self.eps = float(eps)
    super().__init__(params, lr)

  def _init_param_state(self, p: np.ndarray) -> Dict[str, Any]:
    return {"sq_avg": np.zeros_like(p, dtype=float)}

  def _update_param(
      self, p: np.ndarray, g: np.ndarray, state: Dict[str, Any]
  ) -> None:
    sq_avg = state["sq_avg"]
    sq_avg[...] = self.decay_rate * sq_avg + (1.0 - self.decay_rate) * (g**2)
    step_val = g / (np.sqrt(sq_avg) + self.eps)
    delta = (self.lr * step_val).astype(p.dtype, copy=False)
    p[...] -= delta


class AdamOptimizer(BaseOptimizer):
  """Adaptive Moment Estimation (Kingma & Ba)."""

  def __init__(
      self,
      params: Any,
      lr: float = 1e-3,
      beta1: float = 0.9,
      beta2: float = 0.999,
      eps: float = 1e-8,
  ):
    self.beta1 = float(beta1)
    self.beta2 = float(beta2)
    self.eps = float(eps)
    super().__init__(params, lr)

  def _init_param_state(self, p: np.ndarray) -> Dict[str, Any]:
    return {
        "m": np.zeros_like(p, dtype=float),
        "v": np.zeros_like(p, dtype=float),
        "t": 0,
    }

  def _update_param(
      self, p: np.ndarray, g: np.ndarray, state: Dict[str, Any]
  ) -> None:
    state["t"] += 1
    t = state["t"]
    m, v = state["m"], state["v"]

    m[...] = self.beta1 * m + (1.0 - self.beta1) * g
    v[...] = self.beta2 * v + (1.0 - self.beta2) * (g**2)

    # Bias correction
    m_hat = m / (1.0 - self.beta1**t)
    v_hat = v / (1.0 - self.beta2**t)

    step_val = m_hat / (np.sqrt(v_hat) + self.eps)
    delta = (self.lr * step_val).astype(p.dtype, copy=False)
    p[...] -= delta


class OptimizerFactory:
  """Registry-based Creational Factory for optimizers."""

  _registry: Dict[str, Type[BaseOptimizer]] = {}

  @classmethod
  def register(cls, name: str, optimizer_cls: Type[BaseOptimizer]) -> None:
    """Registers a new optimizer class dynamically."""
    cls._registry[name.lower()] = optimizer_cls

  @classmethod
  def create(
      cls, kind: str, params: Any, lr: float = 1e-2, **kwargs
  ) -> BaseOptimizer:
    """Instantiates and returns the requested optimizer product."""
    kind_clean = kind.lower().strip()
    optimizer_cls = cls._registry.get(kind_clean)
    if optimizer_cls is None:
      supported = list(cls._registry.keys())
      raise ValueError(
          f"Unsupported optimizer: '{kind}'. Registered options: {supported}"
      )
    return optimizer_cls(params, lr=lr, **kwargs)

OptimizerFactory.register("sgd", SGDOptimizer)
OptimizerFactory.register("momentum", MomentumOptimizer)
OptimizerFactory.register("rmsprop", RMSpropOptimizer)
OptimizerFactory.register("adam", AdamOptimizer)

def make_optimizer(
    params: Any, lr: float = 1e-2, kind: str = "sgd", **kwargs
) -> Dict[str, Any]:
  """Constructs an optimizer honoring the system contract via Factory Pattern.

  Args:
      params: Arbitrary nested parameter structure (list, dict, array).
      lr: Positive float learning rate.
      kind: Name of algorithm ('sgd', 'momentum', 'rmsprop', 'adam').
      **kwargs: Hyperparameters passed to specific optimizer algorithms.

  Returns:
      Mapping dict with at least {'step': callable(grads) -> None}.
  """
  optimizer_instance = OptimizerFactory.create(
      kind=kind, params=params, lr=lr, **kwargs
  )
  return optimizer_instance.as_contract_dict()

# Step 10 - train_step
def train_step(model, loss_fn, optimizer, x_batch, y_batch):
    """Perform one complete optimization step over a minibatch.

    Inputs:
      model: sequential model dict with 'forward', 'backward', and 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x_batch: np.ndarray of shape (B, D)
      y_batch: np.ndarray of shape (B,) integer class labels

    Returns:
      loss: float, scalar batch loss evaluated BEFORE the parameter update.
      Model parameters are updated in place; shapes unchanged and values finite.
    """
    # TODO: your approach here
    loss, param_grads = forward_backward(model, loss_fn, x_batch, y_batch)

    optimizer["step"](param_grads)

    return float(loss)

# Step 11 - train
def train(model, loss_fn, optimizer, x, y, epochs, batch_size, seed=0):
    """Run a deterministic minibatch training loop.

    Inputs:
      model: sequential model dict with 'forward', 'backward', 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x: np.ndarray of shape (N, D) training features
      y: np.ndarray of shape (N,) integer class labels
      epochs: int, number of full passes over the data
      batch_size: int, minibatch size
      seed: int, RNG seed for deterministic shuffling / batching

    Returns:
      history: list[float] of length `epochs`; history[t] is the mean
      train_step loss over minibatches in epoch t.
      Model parameters are updated in place; shapes unchanged.
    """
    # TODO: your approach here
    n_samples = x.shape[0]

    rng = np.random.default_rng(seed)
    history: list[float] = []

    for _ in range(epochs):
      indices = rng.permutation(n_samples)
      batch_losses: list[float] = []

      for start_idx in range(0, n_samples, batch_size):
        batch_idx = indices[start_idx : start_idx + batch_size]
        x_batch = x[batch_idx]
        y_batch = y[batch_idx]

        loss = train_step(model, loss_fn, optimizer, x_batch, y_batch)
        batch_losses.append(loss)

      epoch_mean_loss = (
          float(np.mean(batch_losses)) if batch_losses else float("nan")
      )
      history.append(epoch_mean_loss)

    return history

# Step 12 - design_network
def design_network(input_dim, num_classes, seed=0):
    """Design and train a net that solves a nonlinear classification task.

    Inputs:
      input_dim: int, feature dimension
      num_classes: int, number of classes
      seed: int, RNG seed for reproducibility

    Returns:
      model: trained sequential model (forward/backward/params)
      metrics: dict with
        'accuracy': float >= 0.90 on an evaluation set,
        'x': np.ndarray (N, input_dim) eval features (N >= 50),
        'y': np.ndarray (N,) integer eval labels.
      The eval set (x, y) must not be linearly separable to high accuracy
      (< 0.82 for a linear classifier), and the model's true accuracy on
      it must match metrics['accuracy'] and be >= 0.90.
    """
    # TODO: your approach here
    rng = np.random.default_rng(seed)

    samples_per_class = max(50, int(np.ceil(120 / num_classes)))
    n_total = samples_per_class * num_classes

    x = np.zeros((n_total, input_dim), dtype=float)
    y = np.zeros(n_total, dtype=int)

    for c in range(num_classes):
      start, end = c * samples_per_class, (c + 1) * samples_per_class
      y[start:end] = c

      if input_dim == 1:
        if c == 0:
          x[start:end, 0] = rng.uniform(-0.35, 0.35, size=samples_per_class)
        else:
          signs = rng.choice([-1.0, 1.0], size=samples_per_class)
          radii = rng.uniform(
              c * 0.9 + 0.2, c * 0.9 + 0.7, size=samples_per_class
          )
          x[start:end, 0] = signs * radii
      else:
        angles = rng.uniform(0.0, 2.0 * np.pi, size=samples_per_class)
        radii = (
            rng.uniform(0.05, 0.45, size=samples_per_class)
            if c == 0
            else rng.uniform(
                c * 0.9 + 0.2, c * 0.9 + 0.7, size=samples_per_class
            )
        )
        x[start:end, 0] = radii * np.cos(angles)
        x[start:end, 1] = radii * np.sin(angles)
        if input_dim > 2:
          x[start:end, 2:] = rng.normal(
              0.0, 0.01, size=(samples_per_class, input_dim - 2)
          )

    hidden_dim = 64
    model = make_sequential([
        make_dense(input_dim, hidden_dim, initialize_weights(input_dim, hidden_dim, scheme='he')),
        make_activation(kind='relu'),
        make_dense(hidden_dim, hidden_dim, initialize_weights(hidden_dim, hidden_dim, scheme='he')),
        make_activation(kind='relu'),
        make_dense(hidden_dim, num_classes, initialize_weights(hidden_dim, num_classes, scheme='he')),
    ])

    loss_fn = make_loss(kind="cross_entropy")
    optimizer = make_optimizer(model["params"], lr=0.03, kind="adam")

    history = train(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        x=x,
        y=y,
        epochs=150,
        batch_size=32,
        seed=seed,
    )

    logits, _ = model["forward"](x)
    acc = float(np.mean(np.argmax(logits, axis=1) == y))

    metrics = {
        "accuracy": acc,
        "x": x,
        "y": y,
    }

    return model, metrics

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

