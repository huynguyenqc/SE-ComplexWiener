import math
import librosa
import numpy as np
import torch
from pydantic import BaseModel
from torch import nn
from torch.nn import functional as F
from typing import List, Optional, Tuple
from deep.modules.conv import Conv1d, ConvTranspose1d
from deep.modules.vq import EMACodebook
from deep.modules.wavenet import WaveNet
from deep.complex_modules.batchnorm import ComplexBatchNorm
from deep.complex_modules.conv import ComplexConv1d
from deep.complex_modules.recurrent import ComplexSplitLSTM
from deep.complex_modules.wavenet import ComplexWaveNet


class MelSpectrogram(nn.Module):
    def __init__(self, n_fft: int, n_filters: int = 128) -> None:
        super(MelSpectrogram, self).__init__()
        self.n_fft: int = n_fft
        self.n_filters: int = n_filters
        W_kf = librosa.filters.mel(
            sr=16000,
            n_fft=self.n_fft,
            n_mels=self.n_filters,
            dtype=np.float32)
        self.register_buffer('W_kf1', torch.from_numpy(W_kf).unsqueeze_(dim=-1))

    def forward(self, Xp_bft: torch.Tensor) -> torch.Tensor:
        """Extract Mel spectrogram

        Args:
            Xp_bft (torch.Tensor): Power spectrogram

        Returns:
            torch.Tensor: Mel (power) spectrogram
        """
        return F.conv1d(Xp_bft, self.W_kf1)


class ComplexCRNN(nn.Module):
    class ConstructorArgs(BaseModel):
        dim: int
        hidden_dim: int
        kernel_size: int
        dilation_list: List[int]
        wavenet_bn_momentum: float = 0.25
        n_stages: int = 1
        pad_mode: str = 'same'
        cond_dim: Optional[int] = None
        bn_momentum_conv_in: float = 0.8
        bn_momentum_conv_out: float = 0.25
        n_rnn_layers: int = 1 
        
    def __init__(
            self, dim: int, 
            hidden_dim: int, 
            kernel_size: int, 
            dilation_list: List[int], 
            wavenet_bn_momentum: float = 0.25, 
            n_stages: int = 1, 
            pad_mode: str = 'same', 
            cond_dim: Optional[int] = None, 
            bn_momentum_conv_in: float = 0.8,
            bn_momentum_conv_out: float = 0.25,
            n_rnn_layers: int = 1
    ) -> None:
        super(ComplexCRNN, self).__init__()

        self.in_layer = nn.Sequential(
            ComplexConv1d(dim, hidden_dim, kernel_size=5, pad_mode=pad_mode),
            ComplexBatchNorm(hidden_dim, momentum=bn_momentum_conv_in),
            nn.PReLU())

        self.wn = ComplexWaveNet(
            residual_dim=hidden_dim, gate_dim=hidden_dim, skip_dim=hidden_dim,
            kernel_size=kernel_size, dilation_list=dilation_list,
            bn_momentum=wavenet_bn_momentum, n_stages=n_stages, pad_mode=pad_mode, 
            cond_dim=cond_dim)

        self.rnn = ComplexSplitLSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, num_layers=n_rnn_layers)

        self.out_layer = nn.Sequential(
            ComplexConv1d(hidden_dim, dim, kernel_size=3, pad_mode=pad_mode),
            ComplexBatchNorm(dim, momentum=bn_momentum_conv_out),
            nn.Tanh())

    def forward(self, x_b2ct: torch.Tensor, c_b2ct: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_b2ct = self.in_layer(x_b2ct)
        x_b2ct = self.wn(x_b2ct, c_b2ct)
        x_b2tc = x_b2ct.transpose(-1, -2)
        x_b2tc = self.rnn(x_b2tc)
        x_b2ct = x_b2tc.transpose(-1, -2)
        x_b2ct = self.out_layer(x_b2ct)
        return x_b2ct
        

class F0ImportanceDistribution(nn.Module):
    def __init__(
            self,
            # f0_min: float = 60.,
            # f0_max: float = 420.,
            # n_f0_candidates: float = 3601,
            # fs: int = 16000,
            # n_fft: int = 512,
            # tau: float = 0.5,
            # theta: float = 5.5
            f0_min: float = 60.,
            f0_max: float = 300.,
            n_f0_candidates: float = 241,
            fs: int = 16000,
            n_fft: int = 512,
            tau: float = 0.45,
            theta: float = 2.0
    ) -> None:
        super(F0ImportanceDistribution, self).__init__()
        self.f0_min: float = f0_min
        self.f0_max: float = f0_max
        self.n_f0_candidates: float = n_f0_candidates
        self.fs: int = fs
        self.f_nyquist: float = fs / 2
        self.n_fft: int = n_fft
        self.n_freq_bins: int = n_fft // 2 + 1
        self.tau: float = tau       # Temperature for softmax distribution
        self.theta: float = theta   # 

        # F0 candidates
        self.register_buffer('q_l', torch.linspace(f0_min, f0_max, n_f0_candidates))

        # Integral matrix
        self.U_qf: np.ndarray = np.zeros((self.n_f0_candidates, self.n_freq_bins))
        for l in range(self.n_f0_candidates):
            last_idx = 0
            for k in range(1, int(math.floor(self.f_nyquist / self.q_l[l])) + 1):
                idx = int(math.floor(self.q_l[l] * k * (self.n_freq_bins - 1) / self.f_nyquist))
                self.U_qf[l, idx] += 1 / math.sqrt(k)
                if idx - last_idx > 1:
                    i = int(math.floor((idx + last_idx) / 2.))
                    if (idx - last_idx) % 2 != 0:
                        self.U_qf[l, i] -= 1 / (2 * math.sqrt(k))
                        self.U_qf[l, i+1] -= 1 / (2 * math.sqrt(k))
                    else:
                        self.U_qf[l, i] -= 1 / math.sqrt(k)
                else:
                    self.U_qf[l, idx] -= 1 / (2 * math.sqrt(k))
                    self.U_qf[l, last_idx] -= 1 / (2 * math.sqrt(k))
                last_idx = idx
        self.register_buffer('U_qf1', torch.from_numpy(self.U_qf[:, :, None].astype(np.float32)))

    def forward(self, logXp_bft: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            logXp_bft: Log power spectrogram
        Returns:
            P_bqt: (Discrete) probability distribution of F0 candidate
            m_bt: Whether or not the frame has harmonic (1: Yes, 0: No)
        """
        # Importance matrix
        Q_bqt = F.conv1d(input=logXp_bft, weight=self.U_qf1)

        # Normalise importance w.r.t. for each sample
        with torch.no_grad():
            muQ_b11 = Q_bqt.mean(dim=(1, 2), keepdim=True).detach()
            sigmaQ_b11 = Q_bqt.std(dim=(1, 2), keepdim=True).detach()
        normQ_bqt = (Q_bqt - muQ_b11) / (sigmaQ_b11 + 1e-5)

        # Distribution of F0
        P_bqt = F.softmax(normQ_bqt / self.tau, dim=1)
        with torch.no_grad():
            # Entropy of each frame
            h_bt = -(P_bqt * (P_bqt + 1e-12).log()).sum(dim=1)
            # The smaller entropy -> the higher confidence (less uniform)
            m_bt = (h_bt < self.theta).float()
        # if 'PROGRAM_DEBUG_MODE' in globals():
        #     if PROGRAM_DEBUG_MODE == True:
        #         return P_bqt, m_bt, Q_bqt, normQ_bqt, h_bt, self.q_l
        return P_bqt, m_bt


class Encoder(nn.Module):
    class ConstructorArgs(BaseModel):
        input_dim: int
        output_dim: int
        residual_dim: int
        gate_dim: int
        skip_dim: int
        kernel_size: int
        dilation_list: List[int]
        bn_momentum: float = 0.25
        n_stages: int = 1
        bn_momentum_conv: float = 0.8
        pad_mode: str = 'same'
        down_sample_factor: int = 2
        cond_dim: Optional[int] = None

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            residual_dim: int,
            gate_dim: int,
            skip_dim: int,
            kernel_size: int,
            dilation_list: List[int],
            bn_momentum: float = 0.25,
            n_stages: int = 1,
            bn_momentum_conv: float = 0.8,
            pad_mode: str = 'same',
            down_sample_factor: int = 2,
            cond_dim: Optional[int] = None) -> None:
        super(Encoder, self).__init__()

        self.in_layer = nn.Sequential(
            Conv1d(input_dim, residual_dim, kernel_size=5, pad_mode=pad_mode),
            nn.BatchNorm1d(residual_dim, momentum=bn_momentum_conv),
            nn.PReLU(residual_dim))

        assert down_sample_factor > 0, 'Down-sampling rate must be positive integer!'
        if down_sample_factor > 1:
            receptive_width = (4 + (down_sample_factor % 2)) * down_sample_factor 

            self.re_sampler = nn.Sequential(
                Conv1d(residual_dim, residual_dim, kernel_size=receptive_width,
                       stride=down_sample_factor, pad_mode=pad_mode),
                nn.BatchNorm1d(residual_dim, momentum=bn_momentum_conv),
                nn.PReLU(residual_dim))
        else:
            self.re_sampler = None

        self.wn = WaveNet(
            residual_dim=residual_dim, gate_dim=gate_dim, skip_dim=skip_dim, 
            kernel_size=kernel_size, dilation_list=dilation_list, bn_momentum=bn_momentum, 
            n_stages=n_stages, pad_mode=pad_mode, cond_dim=cond_dim)

        self.out_layer = nn.Sequential(
            Conv1d(skip_dim, output_dim, kernel_size=kernel_size, pad_mode=pad_mode),
            nn.BatchNorm1d(output_dim, momentum=bn_momentum_conv),
            nn.PReLU(output_dim),
            Conv1d(output_dim, output_dim, kernel_size=1, pad_mode=pad_mode))

    def forward(self, x_bct: torch.Tensor, c_bct: Optional[torch.Tensor] = None) -> torch.Tensor:
        h_bct = self.in_layer(x_bct)
        if self.re_sampler is not None:
            h_bct = self.re_sampler(h_bct)
        h_bct = self.wn(h_bct, c_bct)
        y_bct = self.out_layer(h_bct)
        return y_bct


class Decoder(nn.Module):
    class ConstructorArgs(BaseModel):
        input_dim: int
        output_dim: int
        residual_dim: int
        gate_dim: int
        skip_dim: int
        kernel_size: int
        dilation_list: List[int]
        bn_momentum: float = 0.25
        n_stages: int = 1
        bn_momentum_conv: float = 0.25
        pad_mode: str = 'same'
        up_sample_factor: int = 2
        cond_dim: Optional[int] = None

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            residual_dim: int,
            gate_dim: int,
            skip_dim: int,
            kernel_size: int,
            dilation_list: List[int],
            bn_momentum: float = 0.25,
            n_stages: int = 1,
            bn_momentum_conv: float = 0.25,
            pad_mode: str = 'same',
            up_sample_factor: int = 2,
            cond_dim: Optional[int] = None) -> None:
        super(Decoder, self).__init__()

        self.in_layer = nn.Sequential(
            Conv1d(input_dim, 2*residual_dim, kernel_size=kernel_size, pad_mode=pad_mode),
            nn.BatchNorm1d(2*residual_dim, momentum=bn_momentum_conv),
            nn.GLU(dim=1))

        assert up_sample_factor > 0, 'Down-sampling rate must be positive integer!'
        if up_sample_factor > 1:
            receptive_width = (4 + (up_sample_factor % 2)) * up_sample_factor 

            self.re_sampler = nn.Sequential(
                ConvTranspose1d(residual_dim, 2*residual_dim, kernel_size=receptive_width,
                                stride=up_sample_factor, pad_mode=pad_mode),
                nn.BatchNorm1d(2*residual_dim, momentum=bn_momentum_conv),
                nn.GLU(dim=1))
        else:
            self.re_sampler = None

        self.wn = WaveNet(
            residual_dim=residual_dim, gate_dim=gate_dim, skip_dim=skip_dim,
            kernel_size=kernel_size, dilation_list=dilation_list, bn_momentum=bn_momentum,
            n_stages=n_stages, pad_mode=pad_mode, cond_dim=cond_dim)

        self.out_layer = nn.Sequential(
            Conv1d(skip_dim, 2*skip_dim, kernel_size=kernel_size, pad_mode=pad_mode),
            nn.BatchNorm1d(2*skip_dim, momentum=bn_momentum_conv),
            nn.GLU(dim=1),
            Conv1d(skip_dim, output_dim, kernel_size=15, pad_mode=pad_mode))

    def forward(self, x_bct: torch.Tensor, c_bct: Optional[torch.Tensor] = None) -> torch.Tensor:
        h_bct = self.in_layer(x_bct)
        if self.re_sampler is not None:
            h_bct = self.re_sampler(h_bct)
        h_bct = self.wn(h_bct, c_bct)
        y_bct = self.out_layer(h_bct)
        return y_bct


class UNetEncoder(nn.Module):
    class ConstructorArgs(BaseModel):
        encoder_configs: List[Encoder.ConstructorArgs]
        use_batchnorm: bool = False
        embedding_dims: List[int]

    def __init__(self, **kwargs) -> None:
        super(UNetEncoder, self).__init__()
        configs = self.ConstructorArgs(**kwargs)

        for enc_cfg, dim in zip(configs.encoder_configs, configs.embedding_dims):
            assert enc_cfg.output_dim >= dim, \
                'Embeddding dimension must be smaller than output dimension of encoder!'

        self.embedding_dims: List[int] = configs.embedding_dims
        self.encoders = nn.ModuleList([
            Encoder(**enc_cfg.dict()) for enc_cfg in configs.encoder_configs])
        self.batchnorms = nn.ModuleList([
            (nn.BatchNorm1d(num_features=dim)
             if configs.use_batchnorm
             else nn.Identity()) 
            for dim in self.embedding_dims])
        self.total_stride: int = int(np.prod([
            enc_cfg.down_sample_factor 
            for enc_cfg in configs.encoder_configs]))

    def forward(
            self, 
            x_bct: torch.Tensor, 
            list_c_bct: Optional[List[Optional[torch.Tensor]]] = None
    ) -> List[torch.Tensor]:
        list_z_bct = []
        if list_c_bct is None:
            list_c_bct = [None] * len(self.encoders)
        
        h_bct = x_bct
        for enc_i, dim_i, bn_i, ci_bct in zip(self.encoders, self.embedding_dims, self.batchnorms, list_c_bct):
            h_bct = enc_i(h_bct, ci_bct)

            out_dim_i = h_bct.size(1)  # Output dimension of encoder
            e_bct, h_bct = torch.split(h_bct, split_size_or_sections=[dim_i, out_dim_i - dim_i], dim=1)
            list_z_bct.append(bn_i(e_bct))

        return list_z_bct


class UNetDecoder(nn.Module):
    class ConstructorArgs(BaseModel):
        decoder_configs: List[Decoder.ConstructorArgs]

    def __init__(self, **kwargs) -> None:
        super(UNetDecoder, self).__init__()
        configs = self.ConstructorArgs(**kwargs)
        self.decoders = nn.ModuleList([
            Decoder(**dec_cfg.dict()) for dec_cfg in configs.decoder_configs])

    def forward(
            self,
            list_h_bct: List[torch.Tensor],
            list_c_bct: Optional[List[Optional[torch.Tensor]]] = None
    ) -> torch.Tensor:
        if list_c_bct is None:
            list_c_bct = [None] * len(self.decoders)

        pair_h_bct = []
        for dec_i, hi_bct, ci_bct in zip(self.decoders, list_h_bct, list_c_bct):
            pair_h_bct = pair_h_bct + [hi_bct]
            h_bct = dec_i(torch.cat(pair_h_bct, dim=1), ci_bct)
            pair_h_bct = [h_bct]
        return pair_h_bct[0]


class UNetQuantiserEMA(nn.Module):
    class ConstructorArgs(BaseModel):
        quantiser_configs: List[EMACodebook.ConstructorArgs]
        reservoir_downsampling_rates: Optional[List[int]] = None

    def __init__(self, **kwargs) -> None:
        super(UNetQuantiserEMA, self).__init__()
        configs = self.ConstructorArgs(**kwargs)

        self.quantisers = nn.ModuleList([
            EMACodebook(**vq_cfg.dict())
            for vq_cfg in configs.quantiser_configs])
        self.reservoir_downsampling_rates: List[int] = (
            [1] * len(self.quantisers)
            if configs.reservoir_downsampling_rates is None
            else configs.reservoir_downsampling_rates)

    def update_reservoir(self, list_z_bct: List[torch.Tensor]) -> None:
        for vq_i, r_i, zi_bct in zip(self.quantisers, self.reservoir_downsampling_rates, list_z_bct):
            vq_i.update_reservoir(zi_bct[..., ::r_i].transpose(-1, -2))

    def initialise_codebook_from_reservoir(self) -> None:
        for vq_i in self.quantisers:
            vq_i.initialise_codebook_from_reservoir()

    def set_codebook_ema_momentum(self, lr: Optional[float] = None) -> None:
        if lr is not None:
            for vq_i in self.quantisers:
                vq_i.set_codebook_ema_momentum()
    
    def forward(
            self,
            list_z_bct: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[float]]:
        """
        Args:
            list_z_bct: List of hidden variables
        Return:
            list_zq_bct: List of quantised variables with gradients of list_z_bct
            list_q_bct: List of quantised variables (w.r.t. codebook)
            list_d_bkt: List of distances to each embedding in the codebooks
            list_perplexity: List of perplexity
        """
        list_vq_results = [
            vq_i.quantise_with_logs(zi_bct.transpose(-1, -2))
            for vq_i, zi_bct in zip(self.quantisers, list_z_bct)]
        list_perplexity = [
            logs['perplexity'].item()
            for _, _, _, _, logs in list_vq_results]
        list_q_bct = [
            q_btc.transpose(-1, -2) 
            for _, _, _, q_btc, _ in list_vq_results]
        list_d_bkt = [
            d_btk.transpose(-1, -2) 
            for _, _, d_btk, _, _ in list_vq_results]
        list_zq_bct = [
            zi_bct + (qi_bct - zi_bct).detach()
            for zi_bct, qi_bct in zip(list_z_bct, list_q_bct)]
        
        return list_zq_bct, list_q_bct, list_d_bkt, list_perplexity

        
class VAEUNet(nn.Module):
    class ConstructorArgs(BaseModel):
        encoder_configs: UNetEncoder.ConstructorArgs
        decoder_configs: UNetDecoder.ConstructorArgs

    def __init__(self, **kwargs) -> None:
        super(VAEUNet, self).__init__()

        configs = self.ConstructorArgs(**kwargs)

        self.encoders = UNetEncoder(**configs.encoder_configs.dict())
        self.decoders = UNetDecoder(**configs.decoder_configs.dict())

    @classmethod
    def extract_parameters(
            cls, list_h_bct: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get parameters (mean and log variance) of posterior distribution from hidden variable

        Args:
            list_h_bct (torch.Tensor): List of hidden variables

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: List of (mean, log_var)
        """

        return [h_i_bct.chunk(chunks=2, dim=1) for h_i_bct in list_h_bct]

    def sampling(self, list_ms_bct: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[torch.Tensor]:
        list_z_bct = []
        for mu_i_bct, logSigma2_i_bct in list_ms_bct:
            if self.training:
                z_i_bct = mu_i_bct + (logSigma2_i_bct / 2).exp() * torch.randn_like(mu_i_bct)
            else:
                z_i_bct = mu_i_bct
            list_z_bct.append(z_i_bct)
        return list_z_bct

    def forward(
            self,
            x_bct: torch.Tensor,
            list_cex_bct: Optional[List[Optional[torch.Tensor]]] = None,
            list_cdx_bct: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[
            torch.Tensor,
            List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x_bct: Input
            list_cex_bct: Conditional variables for encoders (same resolution with input)
            list_cdx_bct: Conditional variables for decoders (same resolution with input)
        Return:
            xHat_bct: Reconstructed input
            list_msx_bct: Parameters of latent distribution
        """
        list_hx_bct = self.encoders(x_bct, list_cex_bct)
        list_msx_bct = self.extract_parameters(list_hx_bct)
        list_zx_bct = self.sampling(list_msx_bct)
        xHat_bct = self.decoders(list_zx_bct[::-1], list_cdx_bct)

        return xHat_bct, list_msx_bct


class DenoisingVAEUNet(nn.Module):
    class ConstructorArgs(BaseModel):
        encoder_configs: UNetEncoder.ConstructorArgs
        decoder_configs: UNetDecoder.ConstructorArgs

    def __init__(self, **kwargs) -> None:
        super(DenoisingVAEUNet, self).__init__()

        configs = self.ConstructorArgs(**kwargs)

        self.clean_encoders = UNetEncoder(**configs.encoder_configs.dict())
        self.noisy_encoders = UNetEncoder(**configs.encoder_configs.dict())
        self.decoders = UNetDecoder(**configs.decoder_configs.dict())

    def train(self: 'DenoisingVAEUNet', mode: bool = True) -> 'DenoisingVAEUNet':
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode

        self.clean_encoders.train(mode=False)

        self.noisy_encoders.train(mode=mode)
        self.decoders.train(mode=mode)
        return self

    @classmethod
    def extract_parameters(
            cls, list_h_bct: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get parameters (mean and log variance) of posterior distribution from hidden variable

        Args:
            list_h_bct (torch.Tensor): List of hidden variables

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: List of (mean, log_var)
        """

        return [h_i_bct.chunk(chunks=2, dim=1) for h_i_bct in list_h_bct]

    def sampling(self, list_ms_bct: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[torch.Tensor]:
        list_z_bct = []
        for mu_i_bct, logSigma2_i_bct in list_ms_bct:
            if self.training:
                z_i_bct = mu_i_bct + (logSigma2_i_bct / 2).exp() * torch.randn_like(mu_i_bct)
            else:
                z_i_bct = mu_i_bct
            list_z_bct.append(z_i_bct)
        return list_z_bct

    def forward(
            self,
            x_bct: torch.Tensor,
            y_bct: torch.Tensor,
            list_cex_bct: Optional[List[Optional[torch.Tensor]]] = None,
            list_cdx_bct: Optional[List[Optional[torch.Tensor]]] = None,
            list_cey_bct: Optional[List[Optional[torch.Tensor]]] = None,
            list_cdy_bct: Optional[List[Optional[torch.Tensor]]] = None,
            latent_denoising_mode: bool = False
    ) -> Tuple[
            torch.Tensor,
            List[Tuple[torch.Tensor, torch.Tensor]],
            torch.Tensor,
            List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x_bct: Clean input
            y_bct: Noisy input
            list_cex_bct: Conditional variables for encoders of clean input
            list_cdx_bct: Conditional variables for decoders of clean input
            list_cey_bct: Conditional variables for encoders of noisy input
            list_cdy_bct: Conditional variables for decoders of noisy input
        Return:
            xHatx_bct: Reconstructed input from clean input
            list_msx_bct: Parameters of latent distribution of clean input
            xHaty_bct: Denoised input from noisy input
            list_msy_bct: Parameters of latent distribution of noisy input
        """
        with torch.no_grad():
            list_hx_bct = self.clean_encoders(x_bct, list_cex_bct)
            list_msx_bct = self.extract_parameters(list_hx_bct)
            list_zx_bct = self.sampling(list_msx_bct)

            # Avoid batch-norm to be updated
            current_mode = self.training
            self.decoders.eval()
            xHatx_bct = self.decoders(list_zx_bct[::-1], list_cdx_bct)
            self.decoders.train(current_mode)

        list_hy_bct = self.noisy_encoders(y_bct, list_cey_bct)
        list_msy_bct = self.extract_parameters(list_hy_bct)
        if latent_denoising_mode:
            xHaty_bct = None
        else:
            list_zy_bct = self.sampling(list_msy_bct)
            xHaty_bct = self.decoders(list_zy_bct[::-1], list_cdy_bct)
        # xHaty_bct = self.decoders(
        #     [zy_i_bct.detach() for zy_i_bct in list_zy_bct[::-1]], 
        #     list_cdy_bct)

        return xHatx_bct, list_msx_bct, xHaty_bct, list_msy_bct

    def denoise(
            self,
            y_bct: torch.Tensor,
            list_cey_bct: Optional[List[Optional[torch.Tensor]]] = None,
            list_cdy_bct: Optional[List[Optional[torch.Tensor]]] = None) -> torch.Tensor:
        list_hy_bct = self.noisy_encoders(y_bct, list_cey_bct)
        list_msy_bct = self.extract_parameters(list_hy_bct)
        list_zy_bct = self.sampling(list_msy_bct)
        xHaty_bct = self.decoders(list_zy_bct[::-1], list_cdy_bct)
        return xHaty_bct

        
class AEUNet(nn.Module):
    class ConstructorArgs(BaseModel):
        encoder_configs: UNetEncoder.ConstructorArgs
        decoder_configs: UNetDecoder.ConstructorArgs

    def __init__(self, **kwargs) -> None:
        super(AEUNet, self).__init__()

        configs = self.ConstructorArgs(**kwargs)

        self.encoders = UNetEncoder(**configs.encoder_configs.dict())
        self.decoders = UNetDecoder(**configs.decoder_configs.dict())

    def forward(
            self,
            x_bct: torch.Tensor,
            list_cex_bct: Optional[List[Optional[torch.Tensor]]] = None,
            list_cdx_bct: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_bct: Input
            list_cex_bct: Conditional variables for encoders (same resolution with input)
            list_cdx_bct: Conditional variables for decoders (same resolution with input)
        Return:
            xHat_bct: Reconstructed input
        """
        list_zx_bct = self.encoders(x_bct, list_cex_bct)
        xHat_bct = self.decoders(list_zx_bct[::-1], list_cdx_bct)

        return xHat_bct


class DenoisingAEUNet(nn.Module):
    class ConstructorArgs(BaseModel):
        encoder_configs: UNetEncoder.ConstructorArgs
        decoder_configs: UNetDecoder.ConstructorArgs

    def __init__(self, **kwargs) -> None:
        super(DenoisingAEUNet, self).__init__()

        configs = self.ConstructorArgs(**kwargs)

        self.clean_encoders = UNetEncoder(**configs.encoder_configs.dict())
        self.noisy_encoders = UNetEncoder(**configs.encoder_configs.dict())
        self.decoders = UNetDecoder(**configs.decoder_configs.dict())

    def train(self: 'DenoisingAEUNet', mode: bool = True) -> 'DenoisingAEUNet':
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode

        self.clean_encoders.train(mode=False)

        self.noisy_encoders.train(mode=mode)
        self.decoders.train(mode=mode)
        return self

    def forward(
            self,
            x_bct: torch.Tensor,
            y_bct: torch.Tensor,
            list_cex_bct: Optional[List[Optional[torch.Tensor]]] = None,
            list_cdx_bct: Optional[List[Optional[torch.Tensor]]] = None,
            list_cey_bct: Optional[List[Optional[torch.Tensor]]] = None,
            list_cdy_bct: Optional[List[Optional[torch.Tensor]]] = None
    ) -> Tuple[
            torch.Tensor,
            List[torch.Tensor],
            torch.Tensor,
            List[torch.Tensor]]:
        """
        Args:
            x_bct: Clean input
            y_bct: Noisy input
            list_cex_bct: Conditional variables for encoders of clean input
            list_cdx_bct: Conditional variables for decoders of clean input
            list_cey_bct: Conditional variables for encoders of noisy input
            list_cdy_bct: Conditional variables for decoders of noisy input
        Return:
            xHatx_bct: Reconstructed input from clean input
            list_zx_bct: Hidden variables of clean input
            xHaty_bct: Denoised input from noisy input
            list_zy_bct: Hidden variables of noisy input
        """
        list_zx_bct = self.clean_encoders(x_bct, list_cex_bct)
        xHatx_bct = self.decoders(list_zx_bct[::-1], list_cdx_bct)

        list_zy_bct = self.noisy_encoders(y_bct, list_cey_bct)
        xHaty_bct = self.decoders(list_zy_bct[::-1], list_cdy_bct)

        return xHatx_bct, list_zx_bct, xHaty_bct, list_zy_bct

    def denoise(
            self,
            y_bct: torch.Tensor,
            list_cey_bct: Optional[List[Optional[torch.Tensor]]] = None,
            list_cdy_bct: Optional[List[Optional[torch.Tensor]]] = None) -> torch.Tensor:
        list_zy_bct = self.noisy_encoders(y_bct, list_cey_bct)
        xHaty_bct = self.decoders(list_zy_bct[::-1], list_cdy_bct)
        return xHaty_bct


class VQEMAUNet(nn.Module):
    class ConstructorArgs(BaseModel):
        encoder_configs: UNetEncoder.ConstructorArgs
        decoder_configs: UNetDecoder.ConstructorArgs
        quantiser_configs: UNetQuantiserEMA.ConstructorArgs

    def __init__(self, **kwargs) -> None:
        super(VQEMAUNet, self).__init__()

        configs = self.ConstructorArgs(**kwargs)

        self.encoders = UNetEncoder(**configs.encoder_configs.dict())
        self.decoders = UNetDecoder(**configs.decoder_configs.dict())
        self.quantisers = UNetQuantiserEMA(**configs.quantiser_configs.dict())

    def initialise_codebook_from_reservoir(self):
        self.quantisers.initialise_codebook_from_reservoir()

    def set_codebook_ema_momentum(self, lr: Optional[float] = None) -> None:
        if lr is not None:
            self.quantisers.set_codebook_ema_momentum()

    def forward(
            self,
            x_bct: torch.Tensor,
            list_cex_bct: Optional[List[Optional[torch.Tensor]]] = None,
            list_cdx_bct: Optional[List[Optional[torch.Tensor]]] = None,
            ae_preparing_stage: bool = False,
            do_reservoir_sampling: bool = False
    ) -> Tuple[
            torch.Tensor,
            List[torch.Tensor],
            Optional[List[torch.Tensor]],
            Optional[List[float]]]:
        """
        Args:
            x_bct: Input
            list_cex_bct: Conditional variables for encoders (same resolution with input)
            list_cdx_bct: Conditional variables for decoders (same resolution with input)
            ae_preparing_stage: Whether or not to perform quantisation
        Return:
            xHat_bct: Reconstructed input
            list_zx_bct: Continuous hidden variables
            list_qx_bct: Continuous quantised variables
            list_perplexity: List of perplexity
        """
        list_zx_bct = self.encoders(x_bct, list_cex_bct)
        # Always update reservoir in training stage
        if self.training and do_reservoir_sampling:
            self.quantisers.update_reservoir(list_z_bct=list_zx_bct)

        # AE preparing stage: No need quantisation
        if ae_preparing_stage:
            list_zxq_bct, list_qx_bct, list_perplx_x = list_zx_bct, None, None
        else:
            list_zxq_bct, list_qx_bct, _, list_perplx_x = self.quantisers(list_zx_bct)
        xHat_bct = self.decoders(list_zxq_bct[::-1], list_cdx_bct)

        return xHat_bct, list_zx_bct, list_qx_bct, list_perplx_x

    
class DenoisingVQEMAUNet(nn.Module):
    class ConstructorArgs(BaseModel):
        encoder_configs: UNetEncoder.ConstructorArgs
        decoder_configs: UNetDecoder.ConstructorArgs
        quantiser_configs: UNetQuantiserEMA.ConstructorArgs

    def __init__(self, **kwargs) -> None:
        super(DenoisingVQEMAUNet, self).__init__()

        configs = self.ConstructorArgs(**kwargs)

        self.clean_encoders = UNetEncoder(**configs.encoder_configs.dict())
        self.noisy_encoders = UNetEncoder(**configs.encoder_configs.dict())
        self.quantisers = UNetQuantiserEMA(**configs.quantiser_configs.dict())
        self.decoders = UNetDecoder(**configs.decoder_configs.dict())

        # Only to keep consistent with old code
        self.d_norms = nn.ModuleList([
            nn.BatchNorm2d(num_features=1, affine=False, track_running_stats=True)
            for _ in range(len(self.quantisers.quantisers))])

    def train(self, mode: bool = True) -> 'DenoisingVQEMAUNet':
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode

        self.quantisers.train(mode=False)
        self.clean_encoders.train(mode=False)

        self.noisy_encoders.train(mode=mode)
        self.decoders.train(mode=mode)

        return self

    def forward(
            self,
            x_bct: torch.Tensor,
            y_bct: torch.Tensor,
            list_cex_bct: Optional[List[Optional[torch.Tensor]]] = None,
            list_cdx_bct: Optional[List[Optional[torch.Tensor]]] = None,
            list_cey_bct: Optional[List[Optional[torch.Tensor]]] = None,
            list_cdy_bct: Optional[List[Optional[torch.Tensor]]] = None,
            latent_denoising_mode: bool = False
    ) -> Tuple[
            torch.Tensor,
            List[torch.Tensor],
            List[torch.Tensor],
            torch.Tensor,
            List[torch.Tensor],
            List[torch.Tensor]]:
        """
        Args:
            x_bct: Clean input
            y_bct: Noisy input
            list_cex_bct: Conditional variables for encoders of clean input
            list_cdx_bct: Conditional variables for decoders of clean input
            list_cey_bct: Conditional variables for encoders of noisy input
            list_cdy_bct: Conditional variables for decoders of noisy input
        Return:
            xHatx_bct: Reconstructed input from clean input
            list_zx_bct: Hidden variables of clean input
            list_qx_bct: Quantised variables of clean input
            xHaty_bct: Denoised input from noisy input
            list_zy_bct: Hidden variables of noisy input
            list_qy_bct: Quantised variables of noisy input
        """
        with torch.no_grad():
            list_zx_bct = self.clean_encoders(x_bct, list_cex_bct)
            list_zxq_bct, list_qx_bct, _, _ = self.quantisers(list_zx_bct)

            # Avoid batch-norm to be updated
            current_mode = self.training
            self.decoders.eval()
            xHatx_bct = self.decoders(list_zxq_bct[::-1], list_cdx_bct)
            self.decoders.train(current_mode)

        list_zy_bct = self.noisy_encoders(y_bct, list_cey_bct)
        list_zyq_bct, list_qy_bct, _, _ = self.quantisers(list_zy_bct)

        if latent_denoising_mode:
            xHaty_bct = None
        else:
            xHaty_bct = self.decoders(
                [zyq_i_bct.detach() for zyq_i_bct in list_zyq_bct[::-1]], 
                list_cdy_bct)

        return (
            xHatx_bct, list_zx_bct, list_qx_bct,
            xHaty_bct, list_zy_bct, list_qy_bct)

    def denoise(
            self,
            y_bct: torch.Tensor, 
            list_cey_bct: Optional[List[Optional[torch.Tensor]]] = None,
            list_cdy_bct: Optional[List[Optional[torch.Tensor]]] = None) -> torch.Tensor:

        list_zy_bct = self.noisy_encoders(y_bct, list_cey_bct)
        if self.training:
            list_zyq_bct, _, _, _ = self.quantisers(list_zy_bct)
            xHaty_bct = self.decoders(list_zyq_bct[::-1], list_cdy_bct)
        else:
            xHaty_bct = self.decoders(list_zy_bct[::-1], list_cdy_bct)
        return xHaty_bct

            
class NoiseEstimator(nn.Module):
    class ConstructorArgs(BaseModel):
        speech_encoder_configs: Encoder.ConstructorArgs
        noise_decoder_configs: Decoder.ConstructorArgs
        
    def __init__(self, **kwargs) -> None:
        super(NoiseEstimator, self).__init__()

        configs = self.ConstructorArgs(**kwargs)
        self.speech_encoder = Encoder(**configs.speech_encoder_configs.dict())
        self.noise_decoder = Decoder(**configs.noise_decoder_configs.dict())

    def forward(self, logXpHat_bft: torch.Tensor, logYp_bft: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logXpHat_bft (torch.Tensor): Estimated log variance of clean speech
            logYp_bft (torch.Tensor): Log spectrogram of noisy speech

        Returns:
            torch.Tensor: Estimated log variance of noise
        """
        # Log power inverse of ideal ratio mask
        return self.noise_decoder(
            logYp_bft - logXpHat_bft, 
            self.speech_encoder(logXpHat_bft))
