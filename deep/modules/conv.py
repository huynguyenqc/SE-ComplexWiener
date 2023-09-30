import math
import torch
from pydantic import BaseModel
from torch import nn
from typing import Optional, Tuple, Any


def receiptive_field(kernel_size: int, dilation: int) -> int:
    return dilation * (kernel_size - 1) + 1


def same_pad(kernel_size: int, stride: int, dilation: int) -> int:
    return (receiptive_field(kernel_size, dilation) - stride) // 2


def causal_pad(kernel_size: int, stride: int, dilation: int) -> int:
    return receiptive_field(kernel_size, dilation) - stride


def get_conv_params(pad_mode: Optional[str], 
                    kernel_size: int, stride: int, dilation: int
) -> Tuple[int, Tuple[int, int]]:
    if pad_mode == 'same':
        pad_len: int = same_pad(kernel_size, stride, dilation)
        idx_slice: Tuple[int, int] = (0, 0)
    elif pad_mode == 'causal':
        pad_len: int = causal_pad(kernel_size, stride, dilation)
        idx_slice: Tuple[int, int] = (0, pad_len // stride)
    else:  # Default convolution
        pad_len: int = 0
        idx_slice: Tuple[int, int] = (0, 0)
    return pad_len, idx_slice


def get_deconv_params(pad_mode: Optional[str], 
                    kernel_size: int, stride: int, dilation: int
) -> Tuple[int, Tuple[int, int]]:
    if pad_mode == 'same':
        pad_len: int = same_pad(kernel_size, stride, dilation)
        idx_slice: Tuple[int, int] = (0, 0)
    elif pad_mode == 'causal':
        pad_len: int = 0
        idx_slice: Tuple[int, int] = 0, causal_pad(kernel_size, stride, dilation)
    else:  # Default convolution
        pad_len: int = 0
        idx_slice: Tuple[int, int] = (0, 0)
    return pad_len, idx_slice


class Conv1d(nn.Conv1d):
    class ConstructorArgs(BaseModel):
        in_channels: int
        out_channels: int
        kernel_size: int
        stride: int = 1
        pad_mode: Optional[str] = None
        dilation: int = 1
        groups: int = 1
        bias: bool = True
        device: Any = None
        dtype: Any = None

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            pad_mode: Optional[str] = None,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            device=None,
            dtype=None) -> None:
        
        pad_len, idx_slice = get_conv_params(pad_mode, kernel_size, stride, dilation)
        self._slice: Tuple[int, int] = idx_slice

        super(Conv1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=pad_len,
            padding_mode='zeros',
            device=device,
            dtype=dtype)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        y_bct = super(Conv1d, self).forward(x_bct)
        T = y_bct.size(-1)
        y_bct = y_bct[..., self._slice[0]: T - self._slice[1]]
        return y_bct


class ConvTranspose1d(nn.ConvTranspose1d):
    class ConstructorArgs(BaseModel):
        in_channels: int
        out_channels: int
        kernel_size: int
        stride: int = 1
        pad_mode: Optional[str] = None
        output_padding: int = 0
        groups: int = 1
        bias: bool = True
        dilation: int = 1
        device: Any = None
        dtype: Any = None

    def __init__(
                self,
                in_channels: int,
                out_channels: int,
                kernel_size: int,
                stride: int = 1,
                pad_mode: Optional[str] = None,
                output_padding: int = 0,
                groups: int = 1,
                bias: bool = True,
                dilation: int = 1,
                device=None,
                dtype=None) -> None:
        pad_len, idx_slice = get_deconv_params(pad_mode, kernel_size, stride, dilation)
        self._slice: Tuple[int, int] = idx_slice
            
        super(ConvTranspose1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad_len,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode='zeros',
            device=device,
            dtype=dtype)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        y_bct = super(ConvTranspose1d, self).forward(x_bct)
        T = y_bct.size(-1)
        y_bct = y_bct[..., self._slice[0]: T - self._slice[1]]
        return y_bct


class Conv2d(nn.Conv2d):
    class ConstructorArgs(BaseModel):
        in_channels: int
        out_channels: int
        kernel_size: Tuple[int, int]
        stride: Tuple[int, int] = (1, 1)
        pad_mode: Optional[Tuple[Optional[str], Optional[str]]] = None
        dilation: Tuple[int, int] = (1, 1)
        groups: int = 1
        bias: bool = True
        device: Any = None
        dtype: Any = None

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int, int],
            stride: Tuple[int, int] = (1, 1),
            pad_mode: Optional[str] = None,
            dilation: Tuple[int, int] = (1, 1),
            groups: int = 1,
            bias: bool = True,
            device=None,
            dtype=None) -> None:
        
        if pad_mode is None:
            pad_mode = None, None
        pad_len_0, idx_slice_0 = get_conv_params(pad_mode[0], kernel_size[0], stride[0], dilation[0])
        pad_len_1, idx_slice_1 = get_conv_params(pad_mode[1], kernel_size[1], stride[1], dilation[1])
        self._slice_0: Tuple[int, int] = idx_slice_0
        self._slice_1: Tuple[int, int] = idx_slice_1

        super(Conv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=(pad_len_0, pad_len_1),
            padding_mode='zeros',
            device=device,
            dtype=dtype)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x_bcft: torch.Tensor) -> torch.Tensor:
        y_bcft = super(Conv2d, self).forward(x_bcft)
        T_0 = y_bcft.size(-2)
        T_1 = y_bcft.size(-1)
        y_bcft = y_bcft[..., 
                      self._slice_0[0]: T_0 - self._slice_0[1], 
                      self._slice_1[0]: T_1 - self._slice_1[1]]
        return y_bcft


class ConvTranspose2d(nn.ConvTranspose2d):
    class ConstructorArgs(BaseModel):
        in_channels: int
        out_channels: int
        kernel_size: Tuple[int, int]
        stride: Tuple[int, int] = (1, 1)
        pad_mode: Optional[Tuple[Optional[str], Optional[str]]] = None
        output_padding: Tuple[int, int] = (0, 0)
        groups: int = 1
        bias: bool = True
        dilation: Tuple[int, int] = (1, 1)
        device: Any = None
        dtype: Any = None

    def __init__(
                self,
                in_channels: int,
                out_channels: int,
                kernel_size: Tuple[int, int],
                stride: Tuple[int, int] = (1, 1),
                pad_mode: Optional[Tuple[Optional[str], Optional[str]]] = None,
                output_padding: Tuple[int, int] = 0,
                groups: int = 1,
                bias: bool = True,
                dilation: Tuple[int, int] = (1, 1),
                device=None,
                dtype=None) -> None:
        if pad_mode is None:
            pad_mode = None, None
        pad_len_0, idx_slice_0 = get_deconv_params(pad_mode[0], kernel_size[0], stride[0], dilation[0])
        pad_len_1, idx_slice_1 = get_deconv_params(pad_mode[1], kernel_size[1], stride[1], dilation[1])
        self._slice_0: Tuple[int, int] = idx_slice_0
        self._slice_1: Tuple[int, int] = idx_slice_1
            
        super(ConvTranspose2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(pad_len_0, pad_len_1),
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode='zeros',
            device=device,
            dtype=dtype)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x_bcft: torch.Tensor) -> torch.Tensor:
        y_bcft = super(ConvTranspose2d, self).forward(x_bcft)
        T_0 = y_bcft.size(-2)
        T_1 = y_bcft.size(-1)
        y_bcft = y_bcft[..., 
                        self._slice_0[0]: T_0 - self._slice_0[1],
                        self._slice_1[0]: T_1 - self._slice_1[1]]
        return y_bcft


def sanity_check_conv():
    print('# Same convolution')
    net_configs = Conv1d.ConstructorArgs(**{
        'in_channels': 1,
        'out_channels': 1,
        'kernel_size': 4,
        'stride': 2,
        'pad_mode': 'same'})
    net = Conv1d(**net_configs.dict())
    net.weight.data = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.float32)
    net.bias.data = torch.zeros(1, dtype=torch.float32)
    print('  - Results should be: [[[13, 17]]]')
    x_bct = torch.tensor([[[1, 1, 2, 4]]], dtype=torch.float32)
    y_bct = net(x_bct)
    print('  - Output: ', y_bct.detach().cpu().numpy().tolist())

    print('# Causal convolution')
    net_configs = Conv1d.ConstructorArgs(**{
        'in_channels': 1, 
        'out_channels': 1,
        'kernel_size': 4,
        'stride': 2,
        'pad_mode': 'causal'})
    net = Conv1d(**net_configs.dict())
    net.weight.data = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.float32)
    net.bias.data = torch.zeros(1, dtype=torch.float32)
    print('  - Results should be: [[[7, 25]]]')
    x_bct = torch.tensor([[[1, 1, 2, 4]]], dtype=torch.float32)
    y_bct = net(x_bct)
    print('  - Output: ', y_bct.detach().cpu().numpy().tolist())

    print('# Same deconvolution')
    net_configs = ConvTranspose1d.ConstructorArgs(**{
        'in_channels': 1,
        'out_channels': 1,
        'kernel_size': 4,
        'stride': 2,
        'pad_mode': 'same'})
    net = ConvTranspose1d(**net_configs.dict())
    net.weight.data = torch.tensor([[[-35/289, 1/13, 4/17, 9/169]]], dtype=torch.float32)
    net.bias.data = torch.zeros(1, dtype=torch.float32)
    print('  - Results should be: [[[1, 1, 2, 4]]]')
    x_bct = torch.tensor([[[13, 17]]], dtype=torch.float32)
    y_bct = net(x_bct)
    print('  - Output: ', y_bct.detach().cpu().numpy().tolist())

    print('# Causal deconvolution')
    net_configs = ConvTranspose1d.ConstructorArgs(**{
        'in_channels': 1,
        'out_channels': 1,
        'kernel_size': 4,
        'stride': 2,
        'pad_mode': 'causal'})
    net = ConvTranspose1d(
        in_channels=1,
        out_channels=1,
        kernel_size=4,
        stride=2,
        pad_mode='causal')
    net = ConvTranspose1d(**net_configs.dict())
    net.weight.data = torch.tensor([[[11 / 625, -3 / 625, 2 / 25, 4 / 25]]], dtype=torch.float32)
    net.bias.data = torch.zeros(1, dtype=torch.float32)
    print('  - Results should be: [[[0.0132, -0.0336, 1.0, 1.0]]]')
    x_bct = torch.tensor([[[7, 25]]], dtype=torch.float32)
    y_bct = net(x_bct)
    print('  - Output: ', y_bct.detach().cpu().numpy().tolist())

    print('# 2D causal convolution')
    net_configs = Conv2d.ConstructorArgs(**{
        'in_channels': 1,
        'out_channels': 1,
        'kernel_size': (2, 4),
        'stride': (2, 2),
        'pad_mode': ('same', 'causal')})
    net = Conv2d(**net_configs.dict())
    net.weight.data = torch.tensor([[[[1, 2, 3, 4], [1, 0, -1, 0]]]], dtype=torch.float32)
    net.bias.data = torch.zeros(1, dtype=torch.float32)
    print('  - Results should be: [[[[8, 26]]]]')
    x_bct = torch.tensor([[[[1, 1, 2, 4], [-1, 1, -2, 4]]]], dtype=torch.float32)
    y_bct = net(x_bct)
    print('  - Output: ', y_bct.detach().cpu().numpy().tolist())

    print('# 2D same convolution')
    net_configs = Conv2d.ConstructorArgs(**{
        'in_channels': 1,
        'out_channels': 1,
        'kernel_size': (2, 4),
        'stride': (2, 2),
        'pad_mode': ('same', 'same')})
    net = Conv2d(**net_configs.dict())
    net.weight.data = torch.tensor([[[[1, 2, 3, 4], [1, 0, -1, 0]]]], dtype=torch.float32)
    net.bias.data = torch.zeros(1, dtype=torch.float32)
    print('  - Results should be: [[[[12, 14]]]]')
    x_bct = torch.tensor([[[[1, 1, 2, 4], [-1, 1, -2, 4]]]], dtype=torch.float32)
    y_bct = net(x_bct)
    print('  - Output: ', y_bct.detach().cpu().numpy().tolist())

    print('# 2D causal deconvolution')
    net_configs = ConvTranspose2d.ConstructorArgs(**{
        'in_channels': 1,
        'out_channels': 1,
        'kernel_size': (2, 4),
        'stride': (2, 2),
        'pad_mode': ('same', 'causal')})
    net = ConvTranspose2d(**net_configs.dict())
    net.weight.data = torch.tensor([[[[1, 2, 3, 4], [1, 0, -1, 0]]]], dtype=torch.float32)
    net.bias.data = torch.zeros(1, dtype=torch.float32)
    print('  - Results should be: [[[[12, 24, 50, 76], [12, 0, 2, 0]]]]')
    x_bct = torch.tensor([[[[12, 14]]]], dtype=torch.float32)
    y_bct = net(x_bct)
    print('  - Output: ', y_bct.detach().cpu().numpy().tolist())

    print('# 2D same deconvolution')
    net_configs = ConvTranspose2d.ConstructorArgs(**{
        'in_channels': 1,
        'out_channels': 1,
        'kernel_size': (2, 4),
        'stride': (2, 2),
        'pad_mode': ('same', 'same')})
    net = ConvTranspose2d(**net_configs.dict())
    net.weight.data = torch.tensor([[[[1, 2, 3, 4], [1, 0, -1, 0]]]], dtype=torch.float32)
    net.bias.data = torch.zeros(1, dtype=torch.float32)
    print('  - Results should be: [[[[24, 50, 76, 42], [0, 2, 0, -14]]]]')
    x_bct = torch.tensor([[[[12, 14]]]], dtype=torch.float32)
    y_bct = net(x_bct)
    print('  - Output: ', y_bct.detach().cpu().numpy().tolist())


if __name__ == '__main__':
    sanity_check_conv()
