import numpy as np
from mytorch import Tensor, Dependency

# def sigmoid(x: Tensor) -> Tensor:
#     """
#     TODO: implement sigmoid function
#     hint: you can do it using function you've implemented (not directly define grad func)
#     """
#     sigmoid_result = 1 / (1 + (-x).exp())


#     if x.requires_grad:
#         def grad_fn(grad: np.ndarray):
            
#             s = sigmoid_result.data
#             return grad * s * (1 - s)

        
#         depends_on = [Dependency(x, grad_fn)]
#     else:
#         depends_on = []

#     return Tensor(data=sigmoid_result.data, requires_grad=x.requires_grad, depends_on=depends_on)



def sigmoid(x: Tensor) -> Tensor:
    """
    FIXED: implements sigmoid function using power rule with corrected float argument.
    """
    
    # Step 1: Calculate the denominator (1 + e^-x)
    # Note: We must still use Tensor(1.0) inside the denominator to avoid the previous TypeErrors
    denominator = Tensor(1.0) + (-x).exp()

    # CRITICAL FIX: Pass a simple Python float (-1.0) as the power argument.
    sigmoid_result = denominator ** (-1.0)
    
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