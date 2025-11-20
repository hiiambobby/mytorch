import numpy as np
from mytorch import Tensor, Dependency

def sigmoid(x: Tensor) -> Tensor:
    """
    TODO: implement sigmoid function
    hint: you can do it using function you've implemented (not directly define grad func)
    """
    
    # This automatically creates all necessary intermediate Tensor objects 
    # and their Dependencies.
    sigmoid_result = 1 / (1 + (-x).exp())

    # We rely on the implicit graph for backpropagation.
    # The Tensor returned already has dependencies attached from the operations above.
    return sigmoid_result # Return the Tensor object itself