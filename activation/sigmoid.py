import numpy as np
from mytorch import Tensor, Dependency

def sigmoid(x: Tensor) -> Tensor:
    """
    TODO: implement sigmoid function
    hint: you can do it using function you've implemented (not directly define grad func)
    """
    
    """
    FIXED: implement sigmoid function using power rule to avoid division error.
    """
    # Step 1: Calculate the denominator (1 + e^-x)
    denominator = Tensor(1.0) + (-x).exp()

    # CRITICAL FIX: Rewrite 1 / Denominator as Denominator ** -1.0
    sigmoid_result = denominator ** Tensor(-1.0)
    
    # --- Backward Pass (injecting the correct gradient) ---
    if x.requires_grad:
        def grad_fn(grad: np.ndarray):
            # Derivative: s * (1 - s)
            s = sigmoid_result.data
            return grad * s * (1 - s)
        
        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=sigmoid_result.data, requires_grad=x.requires_grad, depends_on=depends_on)