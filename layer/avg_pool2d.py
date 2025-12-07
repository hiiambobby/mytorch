from mytorch import Tensor, Dependency
from mytorch.layer import Layer
import numpy as np

class AvgPool2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels # معمولاً در پولینگ کانال ورودی و خروجی برابر است
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        KH, KW = self.kernel_size
        SH, SW = self.stride
        PH, PW = self.padding

        # 1. محاسبه ابعاد خروجی
        H_out = (H + 2 * PH - KH) // SH + 1
        W_out = (W + 2 * PW - KW) // SW + 1

        # 2. اعمال پدینگ (اگر نیاز باشد)
        if PH > 0 or PW > 0:
            x_data = np.pad(x.data, ((0,0), (0,0), (PH, PH), (PW, PW)), mode='constant')
        else:
            x_data = x.data

        output = np.zeros((N, C, H_out, W_out))

        # 3. حلقه فوروارد (محاسبه میانگین)
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * SH
                h_end = h_start + KH
                w_start = j * SW
                w_end = w_start + KW
                
                # میانگین‌گیری روی ارتفاع و عرض
                output[:, :, i, j] = np.mean(x_data[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))

        # 4. تعریف عقب‌گرد (Backward Pass)
        requires_grad = x.requires_grad
        depends_on = []

        if requires_grad:
            def grad_fn(grad: np.ndarray):
                # grad shape: (N, C, H_out, W_out)
                grad_input_padded = np.zeros_like(x_data)
                
                # در میانگین‌گیری، گرادیان بر تعداد عناصر (مساحت کرنل) تقسیم می‌شود
                # گرادیان هر پیکسل پنجره = گرادیان خروجی / (ارتفاع_کرنل * عرض_کرنل)
                area = KH * KW
                grad_distributed = grad / area
                
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * SH
                        h_end = h_start + KH
                        w_start = j * SW
                        w_end = w_start + KW
                        
                        # پخش کردن گرادیان روی تمام پنجره ورودی
                        # broadcasting: (N, C) -> (N, C, KH, KW)
                        grad_input_padded[:, :, h_start:h_end, w_start:w_end] += \
                            grad_distributed[:, :, i, j][:, :, None, None]

                # حذف پدینگ برای بازگرداندن گرادیان به سایز اصلی ورودی
                if PH > 0 or PW > 0:
                    return grad_input_padded[:, :, PH:-PH, PW:-PW]
                else:
                    return grad_input_padded

            depends_on.append(Dependency(x, grad_fn))

        return Tensor(data=output, requires_grad=requires_grad, depends_on=depends_on)
    
    def __str__(self) -> str:
        return "avg pool 2d - kernel: {}, stride: {}, padding: {}".format(self.kernel_size, self.stride, self.padding)