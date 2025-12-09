import numpy as np
from mytorch.tensor import Tensor, Dependency


def flatten(x: Tensor) -> Tensor:
    """
    TODO: implement flatten.
    this methods transforms a n dimensional array into a flat array
    hint: use numpy flatten

    Flattens the tensor keeping the batch dimension (Axis 0).
    Input: (N, C, H, W) -> Output: (N, C*H*W)
    """
   
    batch_size = x.shape[0]
    new_shape = (batch_size, -1)
    
    # Flatten
    data = x.data.reshape(new_shape)
    
    requires_grad = x.requires_grad
    depends_on = []

    if requires_grad:
        def grad_fn(grad: np.ndarray):
            # we need to revert to the original shape here
            return grad.reshape(x.shape)
        
        depends_on.append(Dependency(x, grad_fn))

    return Tensor(data=data, requires_grad=requires_grad, depends_on=depends_on)