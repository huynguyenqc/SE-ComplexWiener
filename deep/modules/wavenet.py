import math
import torch
from pydantic import BaseModel
from torch import nn
from typing import List, Optional, Tuple
from deep.modules.conv import Conv1d


class GatedActivationUnit(nn.Module):
    """ WaveNet-like cell """
    class ConstructorArgs(BaseModel):
        residual_dim: int
        gate_dim: int
        skip_dim: int = 128
        kernel_size: int = 3
        dilation: int = 1
        bn_momentum: float = 0.25
        pad_mode: str = 'same'
        with_cond: bool = False

    def __init__(
            self,
            residual_dim: int,
            gate_dim: int,
            skip_dim: int = 128,
            kernel_size: int = 3,
            dilation: int = 1,
            bn_momentum: float = 0.25,
            pad_mode: str = 'same',
            with_cond: bool = False) -> None:
        """ Initialize WNCell module
        Args:
            residual_dim (int): #channels for residual connection.
            gate_dim (int): #channels for gate connection.
            skip_dim (int): #channels for skip connection.
            kernel_size (int): Size of kernel.
            dilation (int): Dilation rate.
            bn_momentum (float): Momentum of batch norm. layers
            pad_mode (str): Padding mode:
                "same": input and output frame length is same,
                "causal": output only depends on current 
                    and previous input frames.
            with_cond (bool): Whether or not to use
                conditional variable.
        """
        super(GatedActivationUnit, self).__init__()
        self.hidden_dim: int = gate_dim
        self.dilation: int = dilation
        self.with_cond: bool = with_cond

        if self.with_cond:
            self.linear_fuse = Conv1d(2*gate_dim, 2*gate_dim,
                                      kernel_size=1, groups=2)

        self.in_layer = nn.Sequential(
            Conv1d(residual_dim, 2 * gate_dim,
                   kernel_size=kernel_size,
                   dilation=dilation,
                   pad_mode=pad_mode),
            nn.BatchNorm1d(2 * gate_dim, momentum=bn_momentum))

        self.res_layer = nn.Sequential(
            Conv1d(gate_dim, residual_dim, kernel_size=kernel_size, pad_mode=pad_mode),
            nn.BatchNorm1d(residual_dim, momentum=bn_momentum))

        self.skip_layer = nn.Sequential(
            Conv1d(gate_dim, skip_dim, kernel_size=kernel_size, pad_mode=pad_mode),
            nn.BatchNorm1d(skip_dim, momentum=bn_momentum))

    def forward(
            self,
            x_bct: torch.Tensor,
            c_bct: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Calculate forward propagation
        Args:
             x_bct (Tensor): input variable
             c_bct (Optional[Tensor]): conditional variable
        Returns:
            Tensor: Output tensor for residual connection 
                (B, residual_channels, T).
            Tensor: Output tensor for skip connection
                (B, skip_channels, T).
        """
        a_bct = self.in_layer(x_bct)

        if self.with_cond:
            assert c_bct is not None
            a_bct = self.linear_fuse(a_bct + c_bct)

        tanh_bct, sigmoid_bct = torch.chunk(a_bct, chunks=2, dim=1)
        a_bct = tanh_bct.tanh() * sigmoid_bct.sigmoid()
        skip_bct = self.skip_layer(a_bct)
        res_bct = self.res_layer(a_bct)
        return (x_bct + res_bct) * math.sqrt(0.5), skip_bct


class WaveNet(nn.Module):
    class ConstructorArgs(BaseModel):
        residual_dim: int
        gate_dim: int
        skip_dim: int
        kernel_size: int
        dilation_list: List[int]
        bn_momentum: float = 0.25
        n_stages: int = 1
        pad_mode: str = 'same'
        cond_dim: Optional[int] = None

    def __init__(
            self,
            residual_dim: int,
            gate_dim: int,
            skip_dim: int,
            kernel_size: int,
            dilation_list: List[int],
            bn_momentum: float = 0.25,
            n_stages: int = 1,
            pad_mode: str = 'same',
            cond_dim: Optional[int] = None) -> None:
        super(WaveNet, self).__init__()

        assert n_stages > 0 and len(dilation_list) > 0
        assert cond_dim is None or cond_dim > 0

        self.units = nn.ModuleList()
        for _ in range(n_stages):
            for d in dilation_list:
                self.units.append(
                    GatedActivationUnit(
                        residual_dim=residual_dim,
                        gate_dim=gate_dim,
                        skip_dim=skip_dim,
                        kernel_size=kernel_size,
                        dilation=d,
                        bn_momentum=bn_momentum,
                        pad_mode=pad_mode,
                        with_cond=cond_dim is not None))

        if cond_dim is not None:
            self.cond_layer = nn.Sequential(
                Conv1d(cond_dim, 2*gate_dim*len(self.units), kernel_size=3, pad_mode=pad_mode),
                nn.BatchNorm1d(2*gate_dim*len(self.units), momentum=bn_momentum),
                nn.PReLU())
        else:
            self.cond_layer = None

    def forward(
            self, 
            x_bct: torch.Tensor,
            c_bct: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if self.cond_layer is not None:
            assert c_bct is not None
            c_bct = self.cond_layer(c_bct)
            list_c_bct = torch.chunk(
                c_bct, chunks=len(self.units), dim=1)
        else:
            list_c_bct = [None] * len(self.units)

        skip_bct = 0
        for gau_i, ci_bct in zip(self.units, list_c_bct):
            x_bct, _skip_bct = gau_i(x_bct, ci_bct)
            skip_bct = skip_bct + _skip_bct
        skip_bct *= math.sqrt(1.0 / len(self.units))
        return skip_bct


def sanity_check():
    from deep import utils
    net = GatedActivationUnit(
        residual_dim=4, gate_dim=6, skip_dim=8, kernel_size=3)
    x_bct = torch.randn(16, 4, 21)
    res_bct, skip_bct = net(x_bct)
    print(x_bct.size(), res_bct.size(), skip_bct.size())

    net = GatedActivationUnit(
        residual_dim=4, gate_dim=6, skip_dim=8, kernel_size=3,
        with_cond=True)
    x_bct, c_bct = torch.randn(16, 4, 21), torch.randn(16, 12, 21)
    res_bct, skip_bct = net(x_bct, c_bct)
    print(x_bct.size(), res_bct.size(), skip_bct.size())

    net = WaveNet(
        residual_dim=4, gate_dim=6, skip_dim=8,
        kernel_size=3, dilation_list=[1, 2, 4])
    x_bct = torch.randn(16, 4, 21)
    y_bct = net(x_bct)
    print(y_bct.size())
    print(utils.count_parameters(net))

    net = WaveNet(
        residual_dim=4, gate_dim=6, skip_dim=8,
        kernel_size=3, dilation_list=[1, 2, 4],
        cond_dim=5)
    x_bct = torch.randn(16, 4, 21)
    c_bct = torch.randn(16, 5, 21)
    y_bct = net(x_bct, c_bct)
    print(y_bct.size())
    print(utils.count_parameters(net))


if __name__ == '__main__':
    sanity_check()
