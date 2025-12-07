from mytorch import Tensor, Dependency
from mytorch.layer import Layer
import numpy as np

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), need_bias: bool = True, mode="xavier") -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.need_bias = need_bias
        self.weight: Tensor = None
        self.bias: Tensor = None
        self.initialize_mode = mode

        self.initialize()
#########TO DOs###########

    def initialize(self):
        # Xavier Initialization
        fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        fan_out = self.out_channels * self.kernel_size[0] * self.kernel_size[1]
        limit = np.sqrt(6 / (fan_in + fan_out))
        
        w_data = np.random.uniform(-limit, limit, (self.out_channels, self.in_channels, *self.kernel_size))
        self.weight = Tensor(data=w_data, requires_grad=True)

        if self.need_bias:
            self.bias = Tensor(data=np.zeros((self.out_channels, 1)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # X shape: (N, C, H, W)
        N, C, H, W = x.shape
        KH, KW = self.kernel_size
        PH, PW = self.padding
        SH, SW = self.stride

        # Calculate output dimensions
        H_out = (H + 2 * PH - KH) // SH + 1
        W_out = (W + 2 * PW - KW) // SW + 1

        # 1. Im2Col Transformation
        col = self._im2col_indices(x.data, H_out, W_out)
        
        # 2. Reshape weights for matrix multiplication
        weights_col = self.weight.data.reshape(self.out_channels, -1)
        
        # 3. Perform Convolution via Matrix Multiplication
        # out_col shape: (out_channels, N * H_out * W_out)
        out_col = weights_col @ col 
        
        if self.need_bias:
            out_col += self.bias.data

        # 4. Reshape back to image (N, out_channels, H_out, W_out)
        out = out_col.reshape(self.out_channels, H_out, W_out, N)
        out = out.transpose(3, 0, 1, 2)

        # 5. Define Backward Pass (Autograd)
        requires_grad = x.requires_grad or self.weight.requires_grad or (self.need_bias and self.bias.requires_grad)
        depends_on = []

        if requires_grad:
            def grad_fn_x(grad: np.ndarray):
                # Gradient w.r.t Input
                grad_reshaped = grad.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
                d_col = weights_col.T @ grad_reshaped
                return self._col2im_indices(d_col, x.shape, H_out, W_out)

            def grad_fn_w(grad: np.ndarray):
                # Gradient w.r.t Weights
                grad_reshaped = grad.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
                dw = grad_reshaped @ col.T
                return dw.reshape(self.weight.shape)

            if x.requires_grad:
                depends_on.append(Dependency(x, grad_fn_x))
            if self.weight.requires_grad:
                depends_on.append(Dependency(self.weight, grad_fn_w))
            
            if self.need_bias and self.bias.requires_grad:
                 def grad_fn_b(grad: np.ndarray):
                    # Sum over N, H, W
                    return np.sum(grad, axis=(0, 2, 3)).reshape(self.bias.shape)
                 depends_on.append(Dependency(self.bias, grad_fn_b))

        return Tensor(data=out, requires_grad=requires_grad, depends_on=depends_on)

    def zero_grad(self):
        if self.weight is not None: self.weight.zero_grad()
        if self.bias is not None: self.bias.zero_grad()

    def parameters(self):
        params = [self.weight]
        if self.need_bias: params.append(self.bias)
        return params

    # --- Helper methods for Im2Col (Speed Optimization) ---
    def _get_im2col_indices(self, x_shape, field_height, field_width, padding=1, stride=1):
        N, C, H, W = x_shape
        assert (H + 2 * padding - field_height) % stride == 0
        assert (W + 2 * padding - field_width) % stride == 0
        out_height = (H + 2 * padding - field_height) // stride + 1
        out_width = (W + 2 * padding - field_width) // stride + 1

        i0 = np.repeat(np.arange(field_height), field_width)
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(field_width), field_height * C)
        j1 = stride * np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
        return (k, i, j)

    def _im2col_indices(self, x, h_out, w_out):
        p = self.padding[0]
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        k, i, j = self._get_im2col_indices(x.shape, self.kernel_size[0], self.kernel_size[1], self.padding[0], self.stride[0])
        cols = x_padded[:, k, i, j]
        C = x.shape[1]
        cols = cols.transpose(1, 2, 0).reshape(self.kernel_size[0] * self.kernel_size[1] * C, -1)
        return cols

    def _col2im_indices(self, cols, x_shape, h_out, w_out):
        N, C, H, W = x_shape
        p = self.padding[0]
        H_padded, W_padded = H + 2 * p, W + 2 * p
        x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
        k, i, j = self._get_im2col_indices(x_shape, self.kernel_size[0], self.kernel_size[1], self.padding[0], self.stride[0])
        cols_reshaped = cols.reshape(C * self.kernel_size[0] * self.kernel_size[1], -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
        if p == 0: return x_padded
        return x_padded[:, :, p:-p, p:-p]

    def __str__(self) -> str:
        return "Conv2d - kernel: {}, stride: {}, padding: {}".format(self.kernel_size, self.stride, self.padding)