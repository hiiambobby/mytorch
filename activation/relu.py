from mytorch import Tensor, Dependency
from mytorch.layer import Layer #this is added to relu for cnn
import numpy as np

class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        data = np.maximum(0, x.data)
        
        req_grad = x.requires_grad
        depends_on = []
        
        if req_grad:
            def grad_fn(grad: np.ndarray):

                return np.where(x.data > 0, grad, 0)
            
            depends_on.append(Dependency(x, grad_fn))
            
        return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
    
    def __str__(self):
        return "ReLU"