import numpy as np
from mytorch import Tensor, Dependency

def sigmoid(x: Tensor) -> Tensor:
    """
    TODO: implement sigmoid function
    hint: you can do it using function you've implemented (not directly define grad func)
    """
    
    # This automatically creates all necessary intermediate Tensor objects 
    # and their Dependencies.
    sigmoid_result = Tensor(1.0) / (Tensor(1.0) + (-x).exp())
    if x.requires_grad:
            def grad_fn(grad: np.ndarray):
                s = sigmoid_result.data
                return grad * s * (1 - s)
            
            depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=sigmoid_result.data, requires_grad=x.requires_grad, depends_on=depends_on)