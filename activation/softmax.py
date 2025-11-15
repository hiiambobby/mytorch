import numpy as np
from mytorch import Tensor, Dependency


def softmax(x: Tensor) -> Tensor:
    """
    TODO: implement softmax function
    hint: you can do it using function you've implemented (not directly define grad func)
    hint: you can't use sum because it has not axis argument so there are 2 ways:
        1. implement sum by axis
        2. using matrix mul to do it :) (recommended)
    hint: a/b = a*(b^-1)
    """
    
    x_exp = x.exp()
    sum_ = x_exp @ Tensor(np.ones((x_exp.shape[-1], 1)))
    out = x_exp * (sum_ ** -1)

    if x.requires_grad:
        def grad_fn(grad: np.ndarray):
            s = out.data
            jacobian = np.diagflat(s) - s @ s.T
            return grad @ jacobian
        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(out.data, requires_grad=x.requires_grad, depends_on=depends_on)