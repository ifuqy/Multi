from tinygrad import Tensor, nn
from typing import List, Callable
from itertools import chain

class BasicUNet:
    def __init__(self, in_channels=1, out_channels=1):
        # Downsampling path (encoder)
        self.down_layers: List[Callable[[Tensor], Tensor]] = [
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ]

        # Upsampling path (decoder)
        self.up_layers: List[Callable[[Tensor], Tensor]] = [
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
        ]

        self.act = lambda x: Tensor.silu(x)  # SiLU activation function
        self.downscale = lambda x: Tensor.max_pool2d(x, 2)
        self.upscale = lambda x: Upsample(scale_factor=2)(x)

    def __call__(self, x: Tensor) -> Tensor:
        h = []
        for i, layer in enumerate(self.down_layers):
            x = self.act(layer(x))  # Convolution + Activation
            if i < 2:
                h.append(x)  # Store skip connection
                x = self.downscale(x)  # Downsampling

        for i, layer in enumerate(self.up_layers):
            if i > 0:
                x = self.upscale(x)  # Upsampling
                x = x + h.pop()  # Add skip connection
            x = self.act(layer(x))  # Convolution + Activation

        return x

class Upsample:
  def __init__(self, scale_factor:int, mode: str = "nearest") -> None:
    assert mode == "nearest" # only mode supported for now
    self.mode = mode
    self.scale_factor = scale_factor

  def __call__(self, x: Tensor) -> Tensor:
    assert len(x.shape) > 2 and len(x.shape) <= 5
    (b, c), _lens = x.shape[:2], len(x.shape[2:])
    tmp = x.reshape([b, c, -1] + [1] * _lens) * Tensor.ones(*[1, 1, 1] + [self.scale_factor] * _lens)
    return tmp.reshape(list(x.shape) + [self.scale_factor] * _lens).permute([0, 1] + list(chain.from_iterable([[y+2, y+2+_lens] for y in range(_lens)]))).reshape([b, c] + [x * self.scale_factor for x in x.shape[2:]])