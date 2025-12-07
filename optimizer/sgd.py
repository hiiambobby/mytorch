from typing import List
from mytorch.layer import Layer
from mytorch.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, layers: List[Layer], learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        "TODO: implement SGD algorithm"

        for layer in self.layers:
            # آپدیت وزن‌ها
            if layer.weight is not None and layer.weight.grad is not None:
                layer.weight.data -= layer.weight.grad.data * self.learning_rate
            
            # آپدیت بایاس (اگر لایه بایاس داشته باشد)
            if layer.need_bias and layer.bias is not None and layer.bias.grad is not None:
                layer.bias.data -= layer.bias.grad.data * self.learning_rate
    
    def zero_grad(self):
        # این تابع هم برای صفر کردن گرادیان‌ها لازم است
        for layer in self.layers:
            layer.zero_grad()