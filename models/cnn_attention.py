from tinygrad import Tensor, nn
from typing import List
import math

class AdaptiveAvgPool2d:
    def __init__(self, output_size):
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def __call__(self, x: Tensor):
        H_in, W_in = x.shape[-2:]
        H_out, W_out = self.output_size

        kernel_h = math.floor(H_in / H_out)
        kernel_w = math.floor(W_in / W_out)

        stride_h = math.floor(H_in / H_out)
        stride_w = math.floor(W_in / W_out)

        return x.avg_pool2d(kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))
    
class AdaptiveMaxPool2d:
    def __init__(self, output_size):
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def __call__(self, x: Tensor):
        H_in, W_in = x.shape[-2:]
        H_out, W_out = self.output_size

        kernel_h = math.floor(H_in / H_out)
        kernel_w = math.floor(W_in / W_out)

        stride_h = math.floor(H_in / H_out)
        stride_w = math.floor(W_in / W_out)

        return x.max_pool2d(kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))


class ChannelAttention:
    def __init__(self, in_planes: int, ratio: int = 16):
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu1 = Tensor.relu
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        
        avg_pool = AdaptiveAvgPool2d(1)(x)
        max_pool = AdaptiveMaxPool2d(1)(x)

        avg_out = self.fc2(self.relu1(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu1(self.fc1(max_pool)))
        return (avg_out + max_out).sigmoid()


class SpatialAttention:
    def __init__(self, kernel_size=3):
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        padding = (kernel_size - 1) // 2
        self.conv_h = nn.Conv2d(2, 1, kernel_size=(kernel_size,1), padding=(padding, 0), bias=False)
        # self.conv_w = nn.Conv2d(2, 1, kernel_size=(1,kernel_size), padding=(0,3), bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        avg_out = x.mean(1, keepdim=True)
        max_out = x.max(1, keepdim=True)
        x_cat = avg_out.cat(max_out, dim=1)

        # attn = self.conv_h(x_cat) + self.conv_w(x_cat)
        attn = self.conv_h(x_cat)
        return attn.sigmoid()



class CNN_Attention:
    def __init__(self, in_channels=1):
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=(5,3), stride=1, padding=(2,1))
        self.bn1 = nn.BatchNorm2d(8)

        self.ca = ChannelAttention(8,8)
        self.sa = SpatialAttention(kernel_size=5)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=(5,3), stride=2, padding=(2,1))
        self.bn2 = nn.BatchNorm2d(16)

        self.pool_01 = lambda x: Tensor.avg_pool2d(x, 2)
        self.pool_02 = lambda x: Tensor.avg_pool2d(x, 2)  

    def _forward_single(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x).relu()

        x = self.pool_01(x)

        x = self.ca(x) * x
        x = self.sa(x) * x

        x = self.conv2(x)
        
        x = self.bn2(x).relu().dropout(0.4)
        x = self.pool_02(x)

        return x.flatten(1)

    def __call__(self, x64: Tensor, x96: Tensor, x128: Tensor) -> Tensor:
        f64 = self._forward_single(x64)
        f96 = self._forward_single(x96)
        f128 = self._forward_single(x128)
        return f64.cat(f96, f128, dim=1)
