from typing import List
from mytorch.layer import Layer
from mytorch.optimizer import Optimizer
import numpy as np


"TODO: (optional) implement Momentum optimizer"

class Momentum(Optimizer):
    def __init__(self, layers: List[Layer], learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = {}

    def step(self):
        for layer in self.layers:
            if layer not in self.velocities:
                self.velocities[layer] = {
                    'w': np.zeros_like(layer.weight.data) if layer.weight is not None else None,
                    'b': np.zeros_like(layer.bias.data) if layer.need_bias and layer.bias is not None else None
                }
            
            if layer.weight is not None and layer.weight.grad is not None:
              
                grad = layer.weight.grad.data + (self.weight_decay * layer.weight.data)
                
                v_old = self.velocities[layer]['w']
                v_new = (self.momentum * v_old) - (self.learning_rate * grad)
                
                self.velocities[layer]['w'] = v_new
                layer.weight.data += v_new
            
            if layer.need_bias and layer.bias is not None and layer.bias.grad is not None:
                grad_b = layer.bias.grad.data
                v_old_b = self.velocities[layer]['b']
                v_new_b = (self.momentum * v_old_b) - (self.learning_rate * grad_b)
                self.velocities[layer]['b'] = v_new_b
                layer.bias.data += v_new_b

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()