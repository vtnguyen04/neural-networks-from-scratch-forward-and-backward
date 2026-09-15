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

# Step 2 - gradient_check (not yet solved)
# TODO: implement

# Step 3 - make_dense (not yet solved)
# TODO: implement

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

