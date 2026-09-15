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

# Step 4 - make_activation (not yet solved)
# TODO: implement

# Step 5 - initialize_weights (not yet solved)
# TODO: implement

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

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

