from mytorch.optimizer import Optimizer
import numpy as np
"TODO: (optional) implement Adam optimizer"

class Adam(Optimizer):
    def __init__(self, layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        self.t = 0
        self.moments = {} 

    def step(self):
        self.t += 1
        for layer in self.layers:
            if layer not in self.moments:
                self.moments[layer] = {
                    'm_w': np.zeros_like(layer.weight.data) if layer.weight is not None else None,
                    'v_w': np.zeros_like(layer.weight.data) if layer.weight is not None else None,
                    'm_b': np.zeros_like(layer.bias.data) if layer.need_bias and layer.bias is not None else None,
                    'v_b': np.zeros_like(layer.bias.data) if layer.need_bias and layer.bias is not None else None
                }

            if layer.weight is not None and layer.weight.grad is not None:
                grad = layer.weight.grad.data + (self.weight_decay * layer.weight.data)
                
                self.moments[layer]['m_w'] = (self.beta1 * self.moments[layer]['m_w']) + \
                                             ((1 - self.beta1) * grad)
                
                self.moments[layer]['v_w'] = (self.beta2 * self.moments[layer]['v_w']) + \
                                             ((1 - self.beta2) * np.square(grad))
                
                m_hat = self.moments[layer]['m_w'] / (1 - self.beta1 ** self.t)
                v_hat = self.moments[layer]['v_w'] / (1 - self.beta2 ** self.t)
                
                layer.weight.data -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon))

            if layer.need_bias and layer.bias is not None and layer.bias.grad is not None:
                grad_b = layer.bias.grad.data
                
                self.moments[layer]['m_b'] = (self.beta1 * self.moments[layer]['m_b']) + \
                                             ((1 - self.beta1) * grad_b)
                
                self.moments[layer]['v_b'] = (self.beta2 * self.moments[layer]['v_b']) + \
                                             ((1 - self.beta2) * np.square(grad_b))
                
                m_hat_b = self.moments[layer]['m_b'] / (1 - self.beta1 ** self.t)
                v_hat_b = self.moments[layer]['v_b'] / (1 - self.beta2 ** self.t)
                
                layer.bias.data -= self.learning_rate * (m_hat_b / (np.sqrt(v_hat_b) + self.epsilon))