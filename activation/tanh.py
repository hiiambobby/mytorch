import numpy as np
from mytorch import Tensor, Dependency

# def tanh(x: Tensor) -> Tensor:
#     """
#     TODO: (optional) implement tanh function
#     hint: you can do it using function you've implemented (not directly define grad func)
#     """
#     exp_x = np.exp(x.data)
#     exp_neg_x = np.exp(-x.data)

#     data = (exp_x - exp_neg_x) / (exp_x + exp_neg_x)

#     req_grad = x.requires_grad

#     if req_grad:
#         def grad_fn(grad: np.ndarray) -> np.ndarray:
#             tanh_x = data
#             return grad * (1 - tanh_x ** 2)

#         depends_on = [Dependency(x, grad_fn)]
#     else:
#         depends_on = []

#     return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)

# Tanh Fix (similar to Sigmoid fix, avoiding / and Tensor arguments to **):
def tanh(x: Tensor) -> Tensor:
    """
    Implements the Hyperbolic Tangent (Tanh) activation function.
    Uses the power rule to ensure numerical stability by avoiding direct division.
    """
    # Calculate components: e^x and e^(-x)
    e_x = x.exp()
    e_minus_x = (-x).exp()

    # Calculate Numerator and Denominator
    numerator = e_x - e_minus_x
    denominator = e_x + e_minus_x

    # CRITICAL FIX: Rewrite division (A / B) using multiplication and power (A * B^-1)
    # This avoids the explicit TypeError for 'Tensor / Tensor'.
    denominator_inv = denominator ** (-1.0)
    tanh_result = numerator * denominator_inv

    if x.requires_grad:
        # The Tanh derivative is d(tanh(x))/dx = 1 - tanh^2(x).
        def grad_fn(grad: np.ndarray): 
            # 's' is the forward result (tanh(x))
            s = tanh_result.data
            # Apply the chain rule: incoming_grad * local_derivative
            # The local derivative is (1 - s^2)
            return grad * (1 - s**2)

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=tanh_result.data, requires_grad=x.requires_grad, depends_on=depends_on)