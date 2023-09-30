import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from scipy import fftpack as scp_fft
from scipy import signal as scp_sig
from typing import Tuple, Optional, Union


def init_rfft_kernels(
        win_len: int,
        n_fft: int) -> Tuple[torch.Tensor, torch.Tensor]:

    N = n_fft
    M = win_len
    K = n_fft // 2 + 1

    Vc_nk = np.fft.rfft(np.eye(N))
    Vr_nk = np.real(Vc_nk)
    Vi_nk = np.imag(Vc_nk)

    # iV_md @ V_dm = eye(M)
    V_dm = np.concatenate(
        (Vr_nk.T[:K, :M], Vi_nk.T[:K, :M]), axis=0)

    c_1d = np.array(
        ([1] + [2] * (K-2) + [1]) * 2
    )[None, :] / N
    iV_md = np.concatenate(
        (Vr_nk[:M, :K], Vi_nk[:M, :K]), axis=1) * c_1d

    return (
        torch.from_numpy(V_dm.astype(np.float32)),
        torch.from_numpy(iV_md.astype(np.float32)))


def init_stft_kernels(
        win_len: int, 
        n_fft: int) -> Tuple[torch.Tensor, torch.Tensor]:

    V_dm, iV_md = init_rfft_kernels(win_len, n_fft)
    V_d1m = V_dm.unsqueeze_(dim=1)
    iV_d1m = iV_md.transpose_(dim0=0, dim1=1).unsqueeze_(dim=1)

    return V_d1m, iV_d1m


def init_dct_kernels(
        n_dct: int) -> Tuple[torch.Tensor, torch.Tensor]:

    V_nn = scp_fft.dct(np.eye(n_dct), type=2, norm='ortho')

    V_qf = V_nn.T
    iV_fq = V_nn
    return (
        torch.from_numpy(V_qf.astype(np.float32)),
        torch.from_numpy(iV_fq.astype(np.float32)))


def init_window(win_len: int, window_type: str) -> torch.Tensor:
    W_m = scp_sig.get_window(
        window=window_type, Nx=win_len, fftbins=True)
    W_11m = W_m[None, None, :]
    return torch.from_numpy(W_11m.astype(np.float32))


def minimum_power_of_two(n: int) -> int:
    return int(2 ** math.ceil(math.log2(n)))


class LinearDCT(nn.Module):
    class ConstructorArgs(BaseModel):
        n_dct: int

    def __init__(self, n_dct: int) -> None:
        super(LinearDCT, self).__init__()
        self.n_dct: int = n_dct
        V_qf, _ = init_dct_kernels(n_dct)
        self.register_buffer(name='V_qf', tensor=V_qf)
    
    def forward(self, x__f: torch.Tensor) -> torch.Tensor:
        assert x__f.size(-1) == self.n_dct
        x__q = F.linear(x__f, self.V_qf)
        return x__q


class LinearIDCT(nn.Module):
    class ConstructorArgs(BaseModel):
        n_dct: int

    def __init__(self, n_dct: int) -> None:
        super(LinearIDCT, self).__init__()

        self.n_dct: int = n_dct
        _, iV_fq = init_dct_kernels(n_dct=n_dct)
        self.register_buffer(name='iV_fq', tensor=iV_fq)
    
    def forward(
            self, 
            x__q: torch.Tensor, 
            q_range_idx: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        assert x__q.size(-1) == self.n_dct

        q_range_idx = q_range_idx or (0, self.n_dct)
        
        x__f = F.linear(x__q[..., q_range_idx[0]: q_range_idx[1]], 
                        self.iV_fq[:, q_range_idx[0]: q_range_idx[1]])
        return x__f


class ConvSTFT(nn.Module):
    class ConstructorArgs(BaseModel):
        win_len: int 
        hop_len: int
        fft_len: Optional[int] = None
        win_type: str = 'hamming'

    def __init__(
            self,
            win_len: int, 
            hop_len: int,
            fft_len: Optional[int] = None,
            win_type: str = 'hamming') -> None:

        super(ConvSTFT, self).__init__()

        self.hop_len: int = hop_len
        self.win_len: int = win_len
        self.fft_len: int = (
            minimum_power_of_two(win_len) if fft_len is None
            else fft_len)
        self.dim: int = self.fft_len // 2 + 1
        self.pad_len: int = self.win_len - self.hop_len

        V_d1m, _ = init_stft_kernels(win_len, self.fft_len)
        W_11m = init_window(win_len, win_type)
        self.register_buffer(name='V_d1m', tensor=V_d1m*W_11m)

    def forward(self, x_bt: torch.Tensor) -> torch.Tensor:
        x_b1t = x_bt.unsqueeze(1)
        x_b1t = F.pad(x_b1t, [self.pad_len, self.pad_len])
        X_bdt = F.conv1d(x_b1t, self.V_d1m, stride=self.hop_len)
        X_b2ft = X_bdt.unflatten(dim=1, sizes=(2, self.dim))

        return X_b2ft


class ConvISTFT(nn.Module):
    class ConstructorArgs(BaseModel):
        win_len: int 
        hop_len: int
        fft_len: Optional[int] = None
        win_type: str = 'hamming'

    def __init__(
            self,
            win_len: int, 
            hop_len: int, 
            fft_len: Optional[int] = None, 
            win_type: str = 'hamming') -> None:

        super(ConvISTFT, self).__init__()

        self.hop_len: int = hop_len
        self.win_len: int = win_len
        self.fft_len: int = (
            minimum_power_of_two(win_len) if fft_len is None
            else fft_len)
        self.dim: int = self.fft_len // 2 + 1
        self.pad_len: int = self.win_len - self.hop_len

        _, iV_d1m = init_stft_kernels(win_len, self.fft_len)
        W_11m = init_window(win_len, win_type)
        I_m1m = torch.from_numpy(
            np.eye(self.win_len, dtype=np.float32)[:, None, :])
        self.register_buffer(name='iV_d1m', tensor=iV_d1m*W_11m)
        self.register_buffer(
            name='W_1m1', tensor=W_11m.transpose(-1, -2))
        self.register_buffer(name='I_m1m', tensor=I_m1m)

    def forward(self, X_b2ft: torch.Tensor) -> torch.Tensor:
        X_bdt = X_b2ft.flatten(start_dim=1, end_dim=2)
        Y_b1t = F.conv_transpose1d(
            X_bdt, self.iV_d1m, stride=self.hop_len)

        W2_1mt = self.W_1m1.square().repeat(1, 1, X_bdt.size(-1))
        c_1mt = F.conv_transpose1d(
            W2_1mt, self.I_m1m, stride=self.hop_len)

        Y_b1t = Y_b1t / (c_1mt + 1e-8)
        Y_b1t = Y_b1t[..., self.pad_len: -self.pad_len]

        return Y_b1t.squeeze(dim=1)