from tinygrad import Tensor, nn

class Bottleneck1D:
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False, expansion: int = 4):
        self.stride = 2 if downsample else 1
        self.expansion = expansion
        self.bottleneck_channels = out_channels // expansion

        # 1x1 Conv for reduction
        self.conv1 = nn.Conv1d(in_channels, self.bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm(self.bottleneck_channels)

        # 3x3 main conv
        self.conv2 = nn.Conv1d(self.bottleneck_channels, self.bottleneck_channels,
                               kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(self.bottleneck_channels)

        # 1x1 Conv for expansion
        self.conv3 = nn.Conv1d(self.bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm(out_channels)

        self.dropout = lambda x: x.dropout(0.4)
        self.act = lambda x: Tensor.relu(x)

        # Residual conv
        if in_channels != out_channels or self.stride != 1:
            self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=self.stride, bias=False)
            self.res_bn = nn.BatchNorm(out_channels)
        else:
            self.res_conv = None

        # SE module
        self.se_fc1 = nn.Conv1d(out_channels, out_channels // 4, kernel_size=1)
        self.se_fc2 = nn.Conv1d(out_channels // 4, out_channels, kernel_size=1)

    def __call__(self, x: Tensor) -> Tensor:
        identity = x

        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)

        out = self.act(self.bn2(self.conv2(out)))
        out = self.dropout(out)

        out = self.bn3(self.conv3(out))

        # SE attention
        se = out.mean(axis=2, keepdim=True)
        se = self.act(self.se_fc1(se))
        se = self.se_fc2(se).sigmoid()
        out = out * se

        # residual path
        if self.res_conv is not None:
            identity = self.res_bn(self.res_conv(identity))

        return self.act(out + identity)

def max_pool1d(x: Tensor, kernel_size: int = 2, stride: int = 2, padding: int = 0) -> Tensor:
    # reshape to (B, C, L, 1)
    x2d = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    # apply 2d pooling with height=kernel, width=1
    out = x2d.max_pool2d(kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(padding, 0))
    # reshape back to (B, C, L')
    return out.reshape(out.shape[0], out.shape[1], out.shape[2])

def avg_pool1d(x: Tensor, kernel_size: int = 2, stride: int = 2, padding: int = 0) -> Tensor:
    # reshape (B, C, L) -> (B, C, L, 1)
    x2d = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    # apply 2D average pooling with kernel on the time dimension only
    out = x2d.avg_pool2d(kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(padding, 0))
    # reshape back to (B, C, L')
    return out.reshape(out.shape[0], out.shape[1], out.shape[2])


class SE_ResNet10_1D:
    def __init__(self, in_channels: int = 1):
        self.conv_init = nn.Conv1d(in_channels, 8, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_init = nn.BatchNorm(8)
        self.pool_init = lambda x: avg_pool1d(x, kernel_size=2, stride=1, padding=1)

        self.block1 = Bottleneck1D(8, 16, expansion=2)
        self.block2 = Bottleneck1D(16, 16, expansion=2)
        self.block3 = Bottleneck1D(16, 32, expansion=2, downsample=True)

        self.pool = lambda x: avg_pool1d(x, kernel_size=2, stride=1, padding=1)
        #self.pool = lambda x: avg_pool1d(x, kernel_size=x.shape[2])
        self.dropout = lambda x: x.dropout(0.4)

    def _forward_branch(self, x: Tensor) -> Tensor:
        x = self.conv_init(x)
        x = self.bn_init(x).relu()
        x = self.pool_init(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # x = self.pool(x)
        return x.flatten(1)

    def __call__(self, x64: Tensor, x96: Tensor, x128: Tensor) -> Tensor:
        f64 = self._forward_branch(x64)
        f96 = self._forward_branch(x96)
        f128 = self._forward_branch(x128)
        return self.dropout(f64.cat(f96, f128, dim=1))
