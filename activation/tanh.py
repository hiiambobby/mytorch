# import numpy as np
# from mytorch import Tensor, Dependency

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
    e_x = x.exp()
    e_minus_x = (-x).exp()
    
    numerator = e_x - e_minus_x  # Tensor - Tensor (Assumes implemented)
    denominator = e_x + e_minus_x # Tensor + Tensor (Assumes implemented)
    
    # CRITICAL FIX: Use power rule to avoid division TypeError
    denominator_inv = denominator ** (-1.0) # Power with float argument
    tanh_result = numerator * denominator_inv # Multiplication
    
    # ... manual backward pass ...
    # grad_fn: (1 - tanh^2) * grad
    s = tanh_result.data
    return grad * (1 - s**2)