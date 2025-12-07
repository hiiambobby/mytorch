from mytorch import Tensor, Dependency
from mytorch.layer import Layer
import numpy as np

class MaxPool2d(Layer):
    def __init__(self, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)) -> None:
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        KH, KW = self.kernel_size
        SH, SW = self.stride
        
        H_out = (H - KH) // SH + 1
        W_out = (W - KW) // SW + 1
        
        # Reshape to facilitate max pooling (naive reshaping method)
        # Note: This version assumes H and W are divisible by kernel for simplicity, or truncates.
        # For a robust implementation similar to Conv2d, im2col is preferred, but simple reshaping works for non-overlapping pools.
        
        reshaped = x.data.reshape(N, C, H_out, SH, W_out, SW)
        out = reshaped.max(axis=(3, 5))
        
        requires_grad = x.requires_grad
        depends_on = []

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                # Gradient of Max is 1 at the max index, 0 otherwise
                grad_input = np.zeros_like(x.data)
                
                # We need to route the gradient to the argmax positions
                # Using repeat to broadcast the gradient back to the window size
                grad_expanded = grad.reshape(N, C, H_out, 1, W_out, 1)
                grad_expanded = np.repeat(grad_expanded, SH, axis=3)
                grad_expanded = np.repeat(grad_expanded, SW, axis=5)
                grad_expanded = grad_expanded.reshape(N, C, H_out*SH, W_out*SW)
                
                # Create mask of max values
                x_reshaped_back = reshaped.reshape(N, C, H_out*SH, W_out*SW)
                mask = (x_reshaped_back == np.repeat(np.repeat(out, SH, axis=2), SW, axis=3))
                
                # Apply mask to gradients (simple approximation handling ties by distributing)
                grad_input[:, :, :H_out*SH, :W_out*SW] = grad_expanded * mask
                return grad_input

            depends_on.append(Dependency(x, grad_fn))

        return Tensor(data=out, requires_grad=requires_grad, depends_on=depends_on)

    def __str__(self) -> str:
        return "MaxPool2d - kernel: {}, stride: {}".format(self.kernel_size, self.stride)