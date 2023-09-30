import math
import random
import torch
from pydantic import BaseModel
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, Literal, List, Optional, Tuple
from deep.modules.fourier import ConvSTFT, ConvISTFT, LinearDCT
from recipies.is2022.modules import (
    Encoder,
    VQEMAUNet, DenoisingVQEMAUNet,
    VAEUNet, DenoisingVAEUNet,
    F0ImportanceDistribution, NoiseEstimator, ComplexCRNN)


class STFTConfigs(BaseModel):
    win_len: int
    hop_len: int
    fft_len: Optional[int] = None
    win_type: str = 'hann'

    
def ceil_div(a: int, b: int) -> int:
    return (a-1) // b + 1


def compute_pad_len_for_inference(
        x__t: torch.Tensor,
        stft_win_len: int,
        stft_hop_len: int,
        required_stride: int = 1,
        look_ahead_frames: int = 0) -> int:
    sig_len = x__t.size(-1)

    n_strided_frames_required = ceil_div(
        (look_ahead_frames - 1) * stft_hop_len + stft_win_len + sig_len,
        required_stride * stft_hop_len)
    n_frames_required = required_stride * n_strided_frames_required
    sig_len_required = (n_frames_required + 1) * stft_hop_len - stft_win_len
    causal_pad_required = sig_len_required - sig_len

    return causal_pad_required

    
def future_estimation(xHat__t: torch.Tensor, look_ahead_width: int = 0) -> torch.Tensor:
    return xHat__t[..., look_ahead_width: ]

    
def past_groundtruth(x__t: torch.Tensor, xHat__t: torch.Tensor) -> torch.Tensor:
    out_len = xHat__t.size(-1)
    return x__t[..., : out_len]


def apply_spectral_augmentation(logXp_bft: torch.Tensor) -> torch.Tensor:
    augmentation_rate = 0.1

    if random.random() > augmentation_rate:
        return logXp_bft

    with torch.no_grad():
        Bb, Ff, Tt = logXp_bft.size()
        neg_inf = -20 * math.log(10)

        n_freqs = 5
        n_times = 3

        ta_1t = torch.arange(n_times, device=logXp_bft.device).unsqueeze_(0)
        fa_1f = torch.arange(n_freqs, device=logXp_bft.device).unsqueeze_(0)
        tm_b1 = torch.randint(Tt-n_times+1, size=(Bb, 1), device=logXp_bft.device)
        fm_b1 = torch.randint(Ff-n_freqs+1, size=(Bb, 1), device=logXp_bft.device)

        mt_b1t = torch.ones(
            Bb, Tt, device=logXp_bft.device, dtype=logXp_bft.dtype
        ).scatter_(dim=1, index=ta_1t+tm_b1, value=neg_inf).unsqueeze_(dim=1)

        mf_bf1 = torch.ones(
            Bb, Ff, device=logXp_bft.device, dtype=logXp_bft.dtype
        ).scatter_(dim=1, index=fa_1f+fm_b1, value=neg_inf).unsqueeze_(dim=2)

        m_bft = mt_b1t * mf_bf1
        return logXp_bft * m_bft    # + neg_inf * (1. - m_bft)


def kl_divergence_log_normal_unit_variance(
        muA__: torch.Tensor,
        muB__: torch.Tensor) -> torch.Tensor:
    """KL divergence between two log normal distributions both of which have variance of 1

    Args:
        muA__ (torch.Tensor): Parameter of the first distribution
        muB__ (torch.Tensor): Parameter of the second distribution

    Returns:
        torch.Tensor: KL divergence between the two distributions
    """
    return 0.5 * F.mse_loss(input=muB__, target=muA__)


def kl_divergence_exponential(
        logSigma2A__: torch.Tensor,
        logSigma2B__: torch.Tensor) -> torch.Tensor:
    """
    KL divergence between two exponential distributions
        Exp(x; lambdaA) and Exp(x; lambdaB)
    
    Args:
        logSigma2A: Parameter of the first distribution (= log(1 / lambdaA))
        logSigma2B: Parameter of the first distribution (= log(1 / lambdaB))
    """
    r__ = logSigma2A__ - logSigma2B__
    return (r__.exp() - r__ - 1).mean()


def kl_divergence_categorical_with_mask(
        P__: torch.Tensor,
        Q__: torch.Tensor,
        m__: torch.Tensor,
        dim: int) -> torch.Tensor:
    """
    KL divergence between two categorical distribution

    Args:
        P__: First categorical distributions
        Q__: Second categorical distributions
        m__: Whether or not the distribution is involved
        dim: Dimension of category
    """
    eps = 1e-6
    return ((P__ * (P__ / (Q__ + eps) + eps).log()).sum(dim=dim) * m__).mean()

    
def complex_mask_multiplication(
        X_b2ft: torch.Tensor,
        M_b2ft: torch.Tensor) -> torch.Tensor:
    Ma_b1kt = (M_b2ft.square().sum(dim=1, keepdim=True) + 1e-5).sqrt()
    M_b2ft = M_b2ft / Ma_b1kt

    Xr_b1ft, Xi_b1ft = X_b2ft.chunk(chunks=2, dim=1)
    Mr_b1ft, Mi_b1ft = M_b2ft.chunk(chunks=2, dim=1)

    # Complex multiplication
    Yr_b1ft = Xr_b1ft * Mr_b1ft - Xi_b1ft * Mi_b1ft
    Yi_b1ft = Xr_b1ft * Mi_b1ft + Xi_b1ft * Mr_b1ft

    Y_b2ft = torch.cat([Yr_b1ft, Yi_b1ft], dim=1)

    return Y_b2ft


def kl_divergence_categorical(
        P__: torch.Tensor, 
        Q__: torch.Tensor, 
        dim: int) -> torch.Tensor:
    """
    KL divergence between two categorical distribution

    Args:
        P__: First categorical distributions
        Q__: Second categorical distributions
        dim: Dimension of category
    """
    eps = 1e-6
    return ((P__ * (P__ / (Q__ + eps) + eps).log()).sum(dim=dim)).mean()


def kl_divergence_normal(
        muA__: torch.Tensor,
        logSig2A__: torch.Tensor,
        muB__: torch.Tensor,
        logSig2B__: torch.Tensor,
        dim: int) -> torch.Tensor:
    """KL divergence between to multivariate normal distributions with diagonal covariance matrices

    Args:
        muA__ (torch.Tensor): Mean of the first distribution
        logSig2A__ (torch.Tensor): Diagon of the covariance matrix of the first distribution
        muB__ (torch.Tensor): Mean of the second distribution
        logSig2B__ (torch.Tensor): Diagon of the covariance matrix of the second distribution
        dim (int): The axis along which represents distribution vector dimension
    """

    logRatioAB__ = logSig2A__ - logSig2B__

    return (
        0.5 * (
            -logRatioAB__ - 1 
            + (muA__ - muB__).square().mul((-logSig2B__).exp()) 
            + logRatioAB__.exp()
        ).sum(dim=dim)
    ).mean()


def negative_signal_to_distortion_ratio_decibel(
        x__t: torch.Tensor,
        xHat__t: torch.Tensor
) -> torch.Tensor:
    """Negative signal-to-distortion ratio in decibel

    Args:
        x__t (torch.Tensor): Target (ground-truth) signal
        xHat__t (torch.Tensor): Estimated signal

    Returns:
        torch.Tensor: Negative signal-to-distortion ration (in dB)
    """
    eps = 1e-5

    alpha__1 = (xHat__t * x__t).mean(dim=-1, keepdim=True) / (
        x__t.square().mean(dim=-1, keepdim=True) + eps)

    x__t = alpha__1 * x__t
    sdr__ = x__t.square().mean(dim=-1) / (
        (xHat__t - x__t).square().mean(dim=-1) + eps)

    sdrdB__ = -10 * (sdr__ + eps).log10()
    return sdrdB__.mean()


class BaseSpeechEnhancement(nn.Module):
    def enhance(self, y_bt: torch.Tensor, phase_correction: bool = False) -> torch.Tensor:
        raise NotImplementedError('This function must be implemented in each subclass')


class PretrainSpeechVariance(nn.Module):
    class ConstructorArgs(BaseModel):
        stft_configs: STFTConfigs
        speech_variance_configs: VQEMAUNet.ConstructorArgs
        cep_encoder_configs: Optional[Encoder.ConstructorArgs] = None
        look_ahead_frames: int = 3
        use_f0_loss: bool = False
        beta_f0: float = 1.0
        spectral_distribution: Literal['exponential', 'log-normal'] = 'exponential'

        # VQVAE training strategy
        preparing_epochs: int = 0
        sampling_epochs: int = 0
        sampling_period: int = 1
        sampling_duration: int = 1

        spectral_augment: bool = False

    def __init__(self, **kwargs) -> None:
        super(PretrainSpeechVariance, self).__init__()

        configs = self.ConstructorArgs(**kwargs)
        self.look_ahead_frames: int = configs.look_ahead_frames
        self.spectral_distribution: Literal['exponential', 'log-normal'] = configs.spectral_distribution

        # STFT and DCt
        self.stft = ConvSTFT(**configs.stft_configs.dict())
        self.istft = ConvISTFT(**configs.stft_configs.dict())
        self.dct = LinearDCT(n_dct=self.stft.dim)
        self.cep_encoder = (
            Encoder(**configs.cep_encoder_configs.dict()) 
            if configs.cep_encoder_configs is not None else None)

        # Speech variance estimator
        self.vqvae = VQEMAUNet(**configs.speech_variance_configs.dict())

        # F0 loss
        self.f0_prob = F0ImportanceDistribution() if configs.use_f0_loss else None
        self.beta_f0 = configs.beta_f0

        # VQVAE training strategy
        self.preparing_epochs: int = configs.preparing_epochs
        self.sampling_epochs: int = configs.sampling_epochs
        self.sampling_period: int = configs.sampling_period
        self.sampling_duration: int = configs.sampling_duration
        self.previous_epoch: int = -1
        self.spectral_augment: bool = configs.spectral_augment

    @property
    def output_logging_keys(self) -> List[str]:
        return (
            ['loss', 'recon_x', 'f0_loss'] 
            + [f'commit_{_k}' for _k in range(len(self.vqvae.encoders.encoders))] 
            + [f'perp_{_k}' for _k in range(len(self.vqvae.encoders.encoders))])

    def forward(self, 
                x_bt: torch.Tensor,
                epoch: int = -1
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        eps = 1e-12
        log_eps = -12 * math.log(10)

        # Autoencoder preparing stages
        ae_preparing_stage = -1 < epoch < self.preparing_epochs

        sampling_idx = (epoch - self.preparing_epochs + self.sampling_duration) // self.sampling_period
        do_reservoir_sampling = (
            sampling_idx >= 0
            and epoch >= self.preparing_epochs + sampling_idx * self.sampling_period - self.sampling_duration
            and epoch < self.preparing_epochs + sampling_idx * self.sampling_period)

        # At the first batch of the epoch
        if self.previous_epoch <= epoch - 1:
            # After preparing stage
            if self.preparing_epochs <= epoch < self.preparing_epochs + self.sampling_epochs:
                if (epoch - self.preparing_epochs) % self.sampling_period == 0:
                    # Initialise codebook from reservoir samples using K-Mean++ algorithm
                    self.vqvae.initialise_codebook_from_reservoir()

        self.previous_epoch = epoch

        with torch.no_grad():
            X_b2ft: torch.Tensor = self.stft(x_bt)
            Xp_bft = X_b2ft.square().sum(dim=1)
            logXp_bft = (Xp_bft + eps).log()

        # Cepstral conditional variable if cepstral encoder is available
        aug_logXp_bft = apply_spectral_augmentation(logXp_bft) if self.spectral_augment else logXp_bft
        list_cdx_bct = None if self.cep_encoder is None else [
            None, self.cep_encoder(
                self.dct(aug_logXp_bft.transpose(-1, -2)).transpose(-1, -2))]

        logXpHat_bft, list_zx_bct, list_qx_bct, list_perp_x = self.vqvae(
            aug_logXp_bft, list_cdx_bct=list_cdx_bct,
            ae_preparing_stage=ae_preparing_stage,
            do_reservoir_sampling=do_reservoir_sampling)
        logXpHat_bft = future_estimation(logXpHat_bft, self.look_ahead_frames).clamp(log_eps, -log_eps)

        # Compute loss
        gamma = 0.0025
        beta = self.beta_f0
        if self.spectral_distribution == 'exponential':
            recon_x = kl_divergence_exponential(past_groundtruth(logXp_bft, logXpHat_bft), logXpHat_bft)
        elif self.spectral_distribution == 'log-normal':
            recon_x = kl_divergence_log_normal_unit_variance(past_groundtruth(logXp_bft, logXpHat_bft), logXpHat_bft)

        if self.f0_prob is not None:
            with torch.no_grad():
                P_bqt, m_bt = self.f0_prob(past_groundtruth(logXp_bft, logXpHat_bft))
            PHat_bqt, _ = self.f0_prob(logXpHat_bft)
            f0_loss = kl_divergence_categorical_with_mask(P_bqt.detach(), PHat_bqt, m_bt.detach(), dim=1)
        else:
            f0_loss = torch.tensor(0.0, device=recon_x.device, dtype=recon_x.dtype)

        if ae_preparing_stage:
            list_commit_x = [
                torch.tensor(0.0, device=x_bt.device, dtype=x_bt.dtype)
                for _ in list_zx_bct]
            list_perp_x = [-1.0 for _ in list_commit_x]
        else:
            list_commit_x = [
                (zx_i_bct - qx_i_bct.detach()).square().sum(dim=1).mean()
                for zx_i_bct, qx_i_bct in zip(list_zx_bct, list_qx_bct)]
        loss = recon_x + beta * f0_loss + gamma * sum(list_commit_x)
        return loss, {
            'loss': loss.item(),
            'recon_x': recon_x.item(),
            'f0_loss': f0_loss.item(),
            **{
                f'commit_{_k}': cl.item() 
                for _k, cl in enumerate(list_commit_x)},
            **{
                f'perp_{_k}': cl
                for _k, cl in enumerate(list_perp_x)}}

    def validate(self, x_bt: torch.Tensor, epoch: int = -1) -> Dict[str, Any]:
        eps = 1e-12
        log_eps = -12 * math.log(10)
        ae_preparing_stage = -1 < epoch < self.preparing_epochs

        pad_len = compute_pad_len_for_inference(
            x_bt, stft_win_len=self.stft.win_len, stft_hop_len=self.stft.hop_len,
            required_stride=self.vqvae.encoders.total_stride,
            look_ahead_frames=self.look_ahead_frames)

        with torch.no_grad():
            x_bt = F.pad(x_bt, pad=(0, pad_len))
            X_b2ft = self.stft(x_bt)
            Xp_bft = X_b2ft.square().sum(dim=1)
            logXp_bft = (Xp_bft + eps).log()

            list_cdx_bct = None if self.cep_encoder is None else [
                None, self.cep_encoder(
                    self.dct(logXp_bft.transpose(-1, -2)).transpose(-1, -2))]

            logXpHat_bft, list_zx_bct, list_qx_bct, list_perp_x = self.vqvae(
                logXp_bft, list_cdx_bct=list_cdx_bct,
                ae_preparing_stage=ae_preparing_stage)
            logXpHat_bft = future_estimation(logXpHat_bft, self.look_ahead_frames).clamp(log_eps, -log_eps)

            # Compute loss
            gamma = 0.0025
            beta = self.beta_f0
            if self.spectral_distribution == 'exponential':
                recon_x = kl_divergence_exponential(past_groundtruth(logXp_bft, logXpHat_bft), logXpHat_bft)
            elif self.spectral_distribution == 'log-normal':
                recon_x = kl_divergence_log_normal_unit_variance(past_groundtruth(logXp_bft, logXpHat_bft), logXpHat_bft)

            if self.f0_prob is not None:
                P_bqt, m_bt = self.f0_prob(past_groundtruth(logXp_bft, logXpHat_bft))
                PHat_bqt, _ = self.f0_prob(logXpHat_bft)
                f0_loss = kl_divergence_categorical_with_mask(P_bqt, PHat_bqt, m_bt, dim=1)
            else:
                f0_loss = torch.tensor(0.0, device=x_bt.device, dtype=recon_x.dtype)

            if ae_preparing_stage:
                list_commit_x = [
                    torch.tensor(0.0, device=x_bt.device, dtype=x_bt.dtype)
                    for _ in list_zx_bct]
                list_perp_x = [-1.0 for _ in list_commit_x]
            else:
                list_commit_x = [
                    (zx_i_bct - qx_i_bct.detach()).square().sum(dim=1).mean()
                    for zx_i_bct, qx_i_bct in zip(list_zx_bct, list_qx_bct)]
            loss = recon_x + beta * f0_loss + gamma * sum(list_commit_x)
            return {
                'spectrum':{
                    'logXp_bft': logXp_bft.detach().cpu(),
                    'logXpHatX_bft': logXpHat_bft.detach().cpu()},
                'numerical': {
                    'loss': loss.item(),
                    'recon_x': recon_x.item(),
                    'f0_loss': f0_loss.item(),
                    **{
                        f'commit_{_k}': cl.item() 
                        for _k, cl in enumerate(list_commit_x)},
                    **{
                        f'perp_{_k}': cl
                        for _k, cl in enumerate(list_perp_x)}}}


class SpeechEnhancement(BaseSpeechEnhancement):
    class ConstructorArgs(BaseModel):
        stft_configs: STFTConfigs
        speech_variance_configs: DenoisingVQEMAUNet.ConstructorArgs
        noise_estimator_configs: NoiseEstimator.ConstructorArgs
        phase_corrector_configs: ComplexCRNN.ConstructorArgs
        cep_encoder_configs: Optional[Encoder.ConstructorArgs] = None
        look_ahead_frames: int = 3
        use_f0_loss: bool = False
        beta_f0: float = 1.0
        spectral_distribution: Literal['exponential', 'log-normal'] = 'exponential'
        use_dkl_loss: bool = True

        wiener_type: Literal['original', 'irm'] = 'irm'

        # Model training strategy
        latent_denoising_epochs: int = 0
        noise_estimation_epochs: int = 0

    def __init__(self, **kwargs) -> None:
        super(SpeechEnhancement, self).__init__()
        configs = self.ConstructorArgs(**kwargs)

        self.use_dkl_loss = configs.use_dkl_loss
        self.look_ahead_frames = configs.look_ahead_frames
        self.spectral_distribution: Literal['exponential', 'log-normal'] = configs.spectral_distribution
        # STFT and DCt
        self.stft = ConvSTFT(**configs.stft_configs.dict())
        self.istft = ConvISTFT(**configs.stft_configs.dict())
        self.dct = LinearDCT(n_dct=self.stft.dim)
        self.cep_clean_encoder = (
            Encoder(**configs.cep_encoder_configs.dict()) 
            if configs.cep_encoder_configs is not None else None)
        self.cep_noisy_encoder = (
            Encoder(**configs.cep_encoder_configs.dict()) 
            if configs.cep_encoder_configs is not None else None)
        
        # Speech variance estimator
        self.d_vqvae = DenoisingVQEMAUNet(**configs.speech_variance_configs.dict())

        # Noise variance estimator
        self.e_noise = NoiseEstimator(**configs.noise_estimator_configs.dict())

        # Phase corrector
        self.phase_corrector = ComplexCRNN(**configs.phase_corrector_configs.dict())

        # F0 loss
        self.f0_prob = F0ImportanceDistribution() if configs.use_f0_loss else None
        self.beta_f0 = configs.beta_f0

        # Wiener type
        self.wiener_type: Literal['original', 'irm'] = configs.wiener_type

        # Training strategy
        self.latent_denoising_epochs: int = configs.latent_denoising_epochs
        self.noise_estimation_epochs: int = configs.noise_estimation_epochs

    @property
    def output_logging_keys(self) -> List[str]:
        return (
            ['loss', 'recon_xY', 'f0_loss_y', 'recon_n', 'condition_loss', 'sisdr_db_xHat1', 'sisdr_db_xHat2', 'sisdr_db_y']
             + [f'qe_{_k}' for _k in range(len(self.d_vqvae.noisy_encoders.encoders))])

    def train(self: 'SpeechEnhancement', mode: bool = True) -> 'SpeechEnhancement':
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode

        self.stft.train(mode=mode)
        self.istft.train(mode=mode)
        if self.cep_noisy_encoder is not None:
            self.dct.train(mode=mode)
            self.cep_clean_encoder.train(mode=False)
            self.cep_noisy_encoder.train(mode=mode)
        self.d_vqvae.train(mode=mode)
        self.e_noise.train(mode=mode)
        self.phase_corrector.train(mode=mode)
        if self.f0_prob is not None:
            self.f0_prob.train(mode=mode)

        return self

    def load_state_dict_from_pretrain_state_dict(
            self,
            pretrain_net_dict: Dict[str, Any]):
        #* The current model (self) and the state dict must be 
        #* on CPU when running this function!
        pretrain_net_configs = PretrainSpeechVariance.ConstructorArgs(**pretrain_net_dict['configs'])
        pretrain_net_state_dict = pretrain_net_dict['state_dict']
        pretrain_net = PretrainSpeechVariance(**pretrain_net_configs.dict())
        pretrain_net.load_state_dict(pretrain_net_state_dict)

        self.d_vqvae.clean_encoders.load_state_dict(pretrain_net.vqvae.encoders.state_dict())
        self.d_vqvae.quantisers.load_state_dict(pretrain_net.vqvae.quantisers.state_dict())
        self.d_vqvae.decoders.load_state_dict(pretrain_net.vqvae.decoders.state_dict())

        if pretrain_net.cep_encoder is not None:
            self.cep_clean_encoder.load_state_dict(pretrain_net.cep_encoder.state_dict())

    def forward(self, x_bt: torch.Tensor, y_bt: torch.Tensor, epoch: int = -1) -> Tuple[torch.Tensor, Dict[str, float]]:
        eps = 1e-12
        log_eps = -12 * math.log(10)
        with torch.no_grad():
            X_b2ft: torch.Tensor = self.stft(x_bt)
            Xp_bft = X_b2ft.square().sum(dim=1)
            logXp_bft = (Xp_bft + eps).log()

            Y_b2ft: torch.Tensor = self.stft(y_bt)
            Yp_bft = Y_b2ft.square().sum(dim=1)
            logYp_bft = (Yp_bft + eps).log()

            n_bt = y_bt - x_bt
            N_b2ft: torch.Tensor = self.stft(n_bt)
            Np_bft = N_b2ft.square().sum(dim=1)
            logNp_bft = (Np_bft + eps).log()

        # Cepstral conditional variable if cepstral encoder is available
        list_cdx_bct = None if self.cep_clean_encoder is None else [
            None, self.cep_clean_encoder(
                self.dct(logXp_bft.transpose(-1, -2)).transpose(-1, -2))]
        list_cdy_bct = None if self.cep_noisy_encoder is None else [
            None, self.cep_noisy_encoder(
                self.dct(logYp_bft.transpose(-1, -2)).transpose(-1, -2))]
        (logXpHatX_bft, _, list_qx_bct,
         logXpHatY_bft, list_zy_bct, _) = self.d_vqvae(
                logXp_bft, logYp_bft,
                list_cdx_bct=list_cdx_bct,
                list_cdy_bct=list_cdy_bct)

        if epoch >= self.latent_denoising_epochs:
            # The autoencoder encodes the input sequence with some additional 
            # future information and decodes the original input sequence.
            logXpHatX_bft = future_estimation(logXpHatX_bft, self.look_ahead_frames).clamp(log_eps, -log_eps)
            logXpHatY_bft = future_estimation(logXpHatY_bft, self.look_ahead_frames).clamp(log_eps, -log_eps)

            # Noise estimation
            logNpHat_bft: torch.Tensor = self.e_noise(
                logXpHatY_bft.detach(),
                past_groundtruth(logYp_bft, logXpHatY_bft))
            logNpHat_bft = future_estimation(logNpHat_bft, self.look_ahead_frames).clamp(log_eps, -log_eps)

            # Ideal ratio mask
            HpHat_bft = 1. / (1. + (logNpHat_bft - past_groundtruth(logXpHatY_bft, logNpHat_bft)).exp())
            if self.wiener_type == 'irm':
                XHat1_b2ft = HpHat_bft.detach().unsqueeze(dim=1).sqrt() * past_groundtruth(Y_b2ft, HpHat_bft)
            elif self.wiener_type == 'original':
                XHat1_b2ft = HpHat_bft.detach().unsqueeze(dim=1) * past_groundtruth(Y_b2ft, HpHat_bft)
            else:
                raise KeyError('Unexpected Wiener type! Must be either "irm" or "original"!')

            # Waveform reconstruction
            with torch.no_grad():
                x_bt = self.istft(X_b2ft)
                y_bt = self.istft(Y_b2ft)
                xHat1_bt = self.istft(XHat1_b2ft)

            if epoch >= self.latent_denoising_epochs + self.noise_estimation_epochs:
                # Phase correction
                MPhiHat_b2ft: torch.Tensor = self.phase_corrector(XHat1_b2ft)
                MPhiHat_b2ft = future_estimation(MPhiHat_b2ft, self.look_ahead_frames)

                XHat2_b2ft = complex_mask_multiplication(
                    past_groundtruth(XHat1_b2ft, MPhiHat_b2ft), MPhiHat_b2ft)

                # Waveform reconstruction
                xHat2_bt = self.istft(XHat2_b2ft)
 
        # -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o- Loss computation -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-
        # --- Speech reconstruction loss
        if epoch >= self.latent_denoising_epochs:
            if self.spectral_distribution == 'exponential':
                recon_xY = kl_divergence_exponential(
                    past_groundtruth(logXp_bft, logXpHatY_bft), logXpHatY_bft)
            elif self.spectral_distribution == 'log-normal':
                recon_xY = kl_divergence_log_normal_unit_variance(
                    past_groundtruth(logXp_bft, logXpHatY_bft), logXpHatY_bft)

            if self.f0_prob is not None:
                with torch.no_grad():
                    P_bqt, m_bt = self.f0_prob(past_groundtruth(logXp_bft, logXpHatY_bft))
                PHatY_bqt, _ = self.f0_prob(logXpHatY_bft)
                f0_loss_y = kl_divergence_categorical_with_mask(P_bqt.detach(), PHatY_bqt, m_bt.detach(), dim=1)
            else:
                f0_loss_y = torch.zeros_like(recon_xY)
        else:
            recon_xY = torch.tensor(0.0).float().to(x_bt.device)
            f0_loss_y = torch.zeros_like(recon_xY)

        # --- Condition loss
        condition_loss = torch.zeros_like(recon_xY) if self.cep_noisy_encoder is None or epoch >= self.latent_denoising_epochs else F.mse_loss(list_cdy_bct[-1], list_cdx_bct[-1])
        
        # --- Quantisation error
        list_qe = [
            (zy_i_bct - qx_i_bct.detach()).square().sum(dim=1).mean()
            for zy_i_bct, qx_i_bct in zip(list_zy_bct, list_qx_bct)]

        # --- Noise reconstruction error
        if epoch >= self.latent_denoising_epochs:
            if self.spectral_distribution == 'exponential':
                recon_n = kl_divergence_exponential(
                    past_groundtruth(logNp_bft, logNpHat_bft), logNpHat_bft)
            elif self.spectral_distribution == 'log-normal':
                recon_n = kl_divergence_log_normal_unit_variance(
                    past_groundtruth(logNp_bft, logNpHat_bft), logNpHat_bft)
        else:
            recon_n = torch.zeros_like(recon_xY)
        
        # --- Negative signal-to-distortion ratio loss for phase reconstruction
        if epoch >= self.latent_denoising_epochs:
            with torch.no_grad():
                nsisdr_db_y = negative_signal_to_distortion_ratio_decibel(x_bt, y_bt)
                nsisdr_db_xHat1 = negative_signal_to_distortion_ratio_decibel(
                    past_groundtruth(x_bt, xHat1_bt), xHat1_bt)
        else:
            nsisdr_db_y = torch.zeros_like(recon_xY)
            nsisdr_db_xHat1 = torch.zeros_like(recon_xY)

        if epoch >= self.latent_denoising_epochs + self.noise_estimation_epochs:
            nsisdr_db_xHat2 = negative_signal_to_distortion_ratio_decibel(
                past_groundtruth(x_bt, xHat2_bt), xHat2_bt)
        else:
            nsisdr_db_xHat2 = torch.zeros_like(recon_xY)

        # Total loss
        beta = self.beta_f0
        loss = (
            recon_xY + beta * f0_loss_y + condition_loss
            + sum(list_qe)
            + recon_n
            + nsisdr_db_xHat2)

        return loss, {
            'loss': loss.item(),
            'recon_xY': recon_xY.item(),
            'f0_loss_y': f0_loss_y.item(),
            'recon_n': recon_n.item(),
            'sisdr_db_xHat1': -nsisdr_db_xHat1.item(),
            'sisdr_db_xHat2': -nsisdr_db_xHat2.item(),
            'sisdr_db_y': -nsisdr_db_y.item(),
            'condition_loss': condition_loss.item(),
            **{
                f'qe_{_k}': cl.item() 
                for _k, cl in enumerate(list_qe)}}

    def validate(self, x_bt: torch.Tensor, y_bt: torch.Tensor, epoch: int = -1) -> Dict[str, Any]:
        eps = 1e-12
        log_eps = -12 * math.log(10)
        pad_len = compute_pad_len_for_inference(
            x_bt, stft_win_len=self.stft.win_len, stft_hop_len=self.stft.hop_len,
            required_stride=self.d_vqvae.clean_encoders.total_stride,
            look_ahead_frames=3*self.look_ahead_frames)

        with torch.no_grad():
            x_bt = F.pad(x_bt, pad=(0, pad_len))
            X_b2ft: torch.Tensor = self.stft(x_bt)
            Xp_bft = X_b2ft.square().sum(dim=1)
            logXp_bft = (Xp_bft + eps).log()

            y_bt = F.pad(y_bt, pad=(0, pad_len))
            Y_b2ft: torch.Tensor = self.stft(y_bt)
            Yp_bft = Y_b2ft.square().sum(dim=1)
            logYp_bft = (Yp_bft + eps).log()

            n_bt = y_bt - x_bt
            N_b2ft: torch.Tensor = self.stft(n_bt)
            Np_bft = N_b2ft.square().sum(dim=1)
            logNp_bft = (Np_bft + eps).log()

            # Cepstral conditional variable if cepstral encoder is available
            list_cdx_bct = None if self.cep_clean_encoder is None else [
                None, self.cep_clean_encoder(
                    self.dct(logXp_bft.transpose(-1, -2)).transpose(-1, -2))]
            list_cdy_bct = None if self.cep_noisy_encoder is None else [
                None, self.cep_noisy_encoder(
                    self.dct(logYp_bft.transpose(-1, -2)).transpose(-1, -2))]
            (logXpHatX_bft, _, list_qx_bct,
             logXpHatY_bft, list_zy_bct, _) = self.d_vqvae(
                    logXp_bft, logYp_bft,
                    list_cdx_bct=list_cdx_bct,
                    list_cdy_bct=list_cdy_bct)

            if epoch >= self.latent_denoising_epochs:
                # The autoencoder encodes the input sequence with some additional 
                # future information and decodes the original input sequence.
                logXpHatX_bft = future_estimation(logXpHatX_bft, self.look_ahead_frames).clamp(log_eps, -log_eps)
                logXpHatY_bft = future_estimation(logXpHatY_bft, self.look_ahead_frames).clamp(log_eps, -log_eps)

                # Noise estimation
                logNpHat_bft: torch.Tensor = self.e_noise(
                    logXpHatY_bft.detach(),
                    past_groundtruth(logYp_bft, logXpHatY_bft))
                logNpHat_bft = future_estimation(logNpHat_bft, self.look_ahead_frames).clamp(log_eps, -log_eps)

                # Ideal ratio mask
                HpHat_bft = 1. / (1. + (logNpHat_bft - past_groundtruth(logXpHatY_bft, logNpHat_bft)).exp())
                if self.wiener_type == 'irm':
                    XHat1_b2ft = HpHat_bft.detach().unsqueeze(dim=1).sqrt() * past_groundtruth(Y_b2ft, HpHat_bft)
                elif self.wiener_type == 'original':
                    XHat1_b2ft = HpHat_bft.detach().unsqueeze(dim=1) * past_groundtruth(Y_b2ft, HpHat_bft)
                else:
                    raise KeyError('Unexpected Wiener type! Must be either "irm" or "original"!')

                # Waveform reconstruction
                with torch.no_grad():
                    x_bt = self.istft(X_b2ft)
                    y_bt = self.istft(Y_b2ft)
                    xHat1_bt = self.istft(XHat1_b2ft)

                if epoch >= self.latent_denoising_epochs + self.noise_estimation_epochs:
                    # Phase correction
                    MPhiHat_b2ft: torch.Tensor = self.phase_corrector(XHat1_b2ft)
                    MPhiHat_b2ft = future_estimation(MPhiHat_b2ft, self.look_ahead_frames)

                    XHat2_b2ft = complex_mask_multiplication(
                        past_groundtruth(XHat1_b2ft, MPhiHat_b2ft), MPhiHat_b2ft)

                    # Waveform reconstruction
                    xHat2_bt = self.istft(XHat2_b2ft)
    
            # -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o- Loss computation -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-
            # --- Speech reconstruction loss
            if epoch >= self.latent_denoising_epochs:
                if self.spectral_distribution == 'exponential':
                    recon_xY = kl_divergence_exponential(
                        past_groundtruth(logXp_bft, logXpHatY_bft), logXpHatY_bft)
                elif self.spectral_distribution == 'log-normal':
                    recon_xY = kl_divergence_log_normal_unit_variance(
                        past_groundtruth(logXp_bft, logXpHatY_bft), logXpHatY_bft)

                if self.f0_prob is not None:
                    with torch.no_grad():
                        P_bqt, m_bt = self.f0_prob(past_groundtruth(logXp_bft, logXpHatY_bft))
                    PHatY_bqt, _ = self.f0_prob(logXpHatY_bft)
                    f0_loss_y = kl_divergence_categorical_with_mask(P_bqt.detach(), PHatY_bqt, m_bt.detach(), dim=1)
                else:
                    f0_loss_y = torch.zeros_like(recon_xY)
            else:
                recon_xY = torch.tensor(0.0).float().to(x_bt.device)
                f0_loss_y = torch.zeros_like(recon_xY)

            # --- Condition loss
            condition_loss = torch.zeros_like(recon_xY) if self.cep_noisy_encoder is None or epoch >= self.latent_denoising_epochs else F.mse_loss(list_cdy_bct[-1], list_cdx_bct[-1])
        
            # --- Quantisation error
            list_qe = [
                (zy_i_bct - qx_i_bct.detach()).square().sum(dim=1).mean()
                for zy_i_bct, qx_i_bct in zip(list_zy_bct, list_qx_bct)]

            # --- Noise reconstruction error
            if epoch >= self.latent_denoising_epochs:
                if self.spectral_distribution == 'exponential':
                    recon_n = kl_divergence_exponential(
                        past_groundtruth(logNp_bft, logNpHat_bft), logNpHat_bft)
                elif self.spectral_distribution == 'log-normal':
                    recon_n = kl_divergence_log_normal_unit_variance(
                        past_groundtruth(logNp_bft, logNpHat_bft), logNpHat_bft)
            else:
                recon_n = torch.zeros_like(recon_xY)
            
            # --- Negative signal-to-distortion ratio loss for phase reconstruction
            if epoch >= self.latent_denoising_epochs:
                with torch.no_grad():
                    nsisdr_db_y = negative_signal_to_distortion_ratio_decibel(x_bt, y_bt)
                    nsisdr_db_xHat1 = negative_signal_to_distortion_ratio_decibel(
                        past_groundtruth(x_bt, xHat1_bt), xHat1_bt)
            else:
                nsisdr_db_y = torch.zeros_like(recon_xY)
                nsisdr_db_xHat1 = torch.zeros_like(recon_xY)

            if epoch >= self.latent_denoising_epochs + self.noise_estimation_epochs:
                nsisdr_db_xHat2 = negative_signal_to_distortion_ratio_decibel(
                    past_groundtruth(x_bt, xHat2_bt), xHat2_bt)
            else:
                nsisdr_db_xHat2 = torch.zeros_like(recon_xY)

            # Total loss
            beta = self.beta_f0
            loss = (
                recon_xY + beta * f0_loss_y + condition_loss
                + sum(list_qe)
                + recon_n
                + nsisdr_db_xHat2)

            return {
                'numerical': {
                    'loss': loss.item(),
                    'recon_xY': recon_xY.item(),
                    'f0_loss_y': f0_loss_y.item(),
                    'recon_n': recon_n.item(),
                    'condition_loss': condition_loss.item(),
                    'sisdr_db_xHat1': -nsisdr_db_xHat1.item(),
                    'sisdr_db_xHat2': -nsisdr_db_xHat2.item(),
                    'sisdr_db_y': -nsisdr_db_y.item(),
                    **{
                        f'qe_{_k}': cl.item() 
                        for _k, cl in enumerate(list_qe)}},
                'waveform': {
                    'x_bt': x_bt.detach().cpu(),
                    'y_bt': y_bt.detach().cpu(),
                    'xHat1_bt': xHat1_bt.detach().cpu(),
                    'xHat2_bt': xHat2_bt.detach().cpu()},
                'spectrum': {
                    'logXp_bft': logXp_bft.detach().cpu(),
                    'logYp_bft': logYp_bft.detach().cpu(),
                    'logNp_bft': logNp_bft.detach().cpu(),
                    'logXpHatX_bft': logXpHatX_bft.detach().cpu(),
                    'logXpHatY_bft': logXpHatY_bft.detach().cpu() if logXpHatY_bft is not None else torch.zeros_like(logXpHatX_bft).detach_().cpu(),
                    'logNpHat_bft': logNpHat_bft.detach().cpu() if logNpHat_bft is not None else torch.zeros_like(logXpHatX_bft).detach_().cpu()},
                'mask': {
                    'HpHat_bft': HpHat_bft.detach().cpu()}}

    def enhance(self, y_bt: torch.Tensor, phase_correction: bool = False) -> torch.Tensor:
        eps = 1e-12
        log_eps = -12 * math.log(10)
        with torch.no_grad():
            # Calculate spectrogram
            Y_b2ft: torch.Tensor = self.stft(y_bt)

            available_len = Y_b2ft.size(-1) - (Y_b2ft.size(-1) % self.d_vqvae.clean_encoders.total_stride)
            Y_b2ft = Y_b2ft[..., : available_len]

            # Log noisy spectrogram
            logYp_bft: torch.Tensor = (Y_b2ft.square().sum(dim=1) + eps).log()

            # Estimated log variance of speech
            list_cdy_bct = None if self.cep_noisy_encoder is None else [
                None, self.cep_noisy_encoder(
                    self.dct(logYp_bft.transpose(-1, -2)).transpose(-1, -2))]
            logXpHatY_bft: torch.Tensor = self.d_vqvae.denoise(logYp_bft, list_cdy_bct=list_cdy_bct)
            logXpHatY_bft = future_estimation(logXpHatY_bft, self.look_ahead_frames).clamp(log_eps, -log_eps)

            # Noise variance estimation
            logNpHat_bft: torch.Tensor = self.e_noise(
                logXpHatY_bft, 
                past_groundtruth(logYp_bft, logXpHatY_bft))
            logNpHat_bft = future_estimation(logNpHat_bft, self.look_ahead_frames).clamp(log_eps, -log_eps)

            # Ideal ratio mask from Wiener filter
            HpHat_bft = 1. / (1. + (logNpHat_bft - past_groundtruth(logXpHatY_bft, logNpHat_bft)).exp())

            if self.wiener_type == 'irm':
                XHat_b2ft = HpHat_bft.unsqueeze(dim=1).sqrt() * past_groundtruth(Y_b2ft, HpHat_bft)
            elif self.wiener_type == 'original':
                XHat_b2ft = HpHat_bft.unsqueeze(dim=1) * past_groundtruth(Y_b2ft, HpHat_bft)
            else:
                raise KeyError('Unexpected Wiener type! Must be either "irm" or "original"!')

            # Phase correction
            if phase_correction:
                MPhiHat_b2ft: torch.Tensor = self.phase_corrector(XHat_b2ft)
                MPhiHat_b2ft = future_estimation(MPhiHat_b2ft, self.look_ahead_frames)

                XHat_b2ft = complex_mask_multiplication(
                    past_groundtruth(XHat_b2ft, MPhiHat_b2ft),
                    MPhiHat_b2ft)

            xHat_bt: torch.Tensor = self.istft(XHat_b2ft)

            return xHat_bt


class PretrainSpeechVarianceVAE(nn.Module):
    class ConstructorArgs(BaseModel):
        stft_configs: STFTConfigs
        speech_variance_configs: VAEUNet.ConstructorArgs
        look_ahead_frames: int = 3
        cep_encoder_configs: Optional[Encoder.ConstructorArgs] = None
        spectral_distribution: Literal['exponential', 'log-normal'] = 'exponential'

    def __init__(self, **kwargs) -> None:
        super(PretrainSpeechVarianceVAE, self).__init__()

        configs = self.ConstructorArgs(**kwargs)
        self.look_ahead_frames: int = configs.look_ahead_frames
        self.spectral_distribution: Literal['exponential', 'log-normal'] = configs.spectral_distribution

        # STFT and DCt
        self.stft = ConvSTFT(**configs.stft_configs.dict())
        self.istft = ConvISTFT(**configs.stft_configs.dict())
        self.dct = LinearDCT(n_dct=self.stft.dim)
        self.cep_encoder = (
            Encoder(**configs.cep_encoder_configs.dict()) 
            if configs.cep_encoder_configs is not None else None)

        # Speech variance estimator
        self.vae = VAEUNet(**configs.speech_variance_configs.dict())

    @property
    def output_logging_keys(self) -> List[str]:
        return (
            ['loss', 'recon_x']
            + [f'regularisation_{_k}' for _k in range(len(self.vae.encoders.encoders))])

    def forward(self, 
                x_bt: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        eps = 1e-12
        log_eps = -12 * math.log(10)

        with torch.no_grad():
            X_b2ft = self.stft(x_bt)
            Xp_bft = X_b2ft.square().sum(dim=1)
            logXp_bft = (Xp_bft + eps).log()

        # Cepstral conditional variable if cepstral encoder is available
        list_cdx_bct = None if self.cep_encoder is None else [
            None, self.cep_encoder(
                self.dct(logXp_bft.transpose(-1, -2)).transpose(-1, -2))]

        logXpHat_bft, list_msx_bct = self.vae(logXp_bft, list_cdx_bct=list_cdx_bct)
        logXpHat_bft = future_estimation(logXpHat_bft, self.look_ahead_frames).clamp(log_eps, -log_eps)

        # Compute loss
        if self.spectral_distribution == 'exponential':
            recon_x = kl_divergence_exponential(past_groundtruth(logXp_bft, logXpHat_bft), logXpHat_bft)
        elif self.spectral_distribution == 'log-normal':
            recon_x = kl_divergence_log_normal_unit_variance(past_groundtruth(logXp_bft, logXpHat_bft), logXpHat_bft)
        list_regularisation = [
            kl_divergence_normal(
                muA__=mu_i_bct,
                logSig2A__=logSigma2_i_bct,
                muB__=torch.zeros_like(mu_i_bct),
                logSig2B__=torch.zeros_like(logSigma2_i_bct),
                dim=1)
            for (mu_i_bct, logSigma2_i_bct) in list_msx_bct]
        loss = recon_x + sum(list_regularisation)
        return loss, {
            'loss': loss.item(),
            'recon_x': recon_x.item(),
            **{
                f'regularisation_{_k}': cl.item() 
                for _k, cl in enumerate(list_regularisation)}}

    def validate(self, x_bt: torch.Tensor) -> Dict[str, Any]:
        eps = 1e-12
        log_eps = -12 * math.log(10)
        pad_len = compute_pad_len_for_inference(
            x_bt, stft_win_len=self.stft.win_len, stft_hop_len=self.stft.hop_len,
            required_stride=self.vae.encoders.total_stride,
            look_ahead_frames=self.look_ahead_frames)

        with torch.no_grad():
            x_bt = F.pad(x_bt, pad=(0, pad_len))
            X_b2ft = self.stft(x_bt)
            Xp_bft = X_b2ft.square().sum(dim=1)
            logXp_bft = (Xp_bft + eps).log()

            # Cepstral conditional variable if cepstral encoder is available
            list_cdx_bct = None if self.cep_encoder is None else [
                None, self.cep_encoder(
                    self.dct(logXp_bft.transpose(-1, -2)).transpose(-1, -2))]

            logXpHat_bft, list_msx_bct = self.vae(logXp_bft, list_cdx_bct=list_cdx_bct)
            logXpHat_bft = future_estimation(logXpHat_bft, self.look_ahead_frames).clamp(log_eps, -log_eps)

            # Compute loss
            if self.spectral_distribution == 'exponential':
                recon_x = kl_divergence_exponential(past_groundtruth(logXp_bft, logXpHat_bft), logXpHat_bft)
            elif self.spectral_distribution == 'log-normal':
                recon_x = kl_divergence_log_normal_unit_variance(past_groundtruth(logXp_bft, logXpHat_bft), logXpHat_bft)
            list_regularisation = [
                kl_divergence_normal(
                    muA__=mu_i_bct, 
                    logSig2A__=logSigma2_i_bct,
                    muB__=torch.zeros_like(mu_i_bct),
                    logSig2B__=torch.zeros_like(logSigma2_i_bct),
                    dim=1)
                for (mu_i_bct, logSigma2_i_bct) in list_msx_bct]
            loss = recon_x + sum(list_regularisation)

            return {
                'spectrum':{
                    'logXp_bft': logXp_bft.detach().cpu(),
                    'logXpHatX_bft': logXpHat_bft.detach().cpu()},
                'numerical': {
                    'loss': loss.item(),
                    'recon_x': recon_x.item(),
                    **{
                        f'regularisation_{_k}': cl.item() 
                        for _k, cl in enumerate(list_regularisation)}}}


class SpeechEnhancementVAE(BaseSpeechEnhancement):
    class ConstructorArgs(BaseModel):
        stft_configs: STFTConfigs
        speech_variance_configs: DenoisingVAEUNet.ConstructorArgs
        noise_estimator_configs: NoiseEstimator.ConstructorArgs
        phase_corrector_configs: ComplexCRNN.ConstructorArgs
        cep_encoder_configs: Optional[Encoder.ConstructorArgs] = None
        look_ahead_frames: int = 3

        spectral_distribution: Literal['exponential', 'log-normal'] = 'exponential'

        wiener_type: Literal['original', 'irm'] = 'irm'

        # Model training strategy
        latent_denoising_epochs: int = 0
        noise_estimation_epochs: int = 0

    def __init__(self, **kwargs) -> None:
        super(SpeechEnhancementVAE, self).__init__()
        configs = self.ConstructorArgs(**kwargs)

        self.look_ahead_frames = configs.look_ahead_frames
        self.spectral_distribution: Literal['exponential', 'log-normal'] = configs.spectral_distribution
        # STFT and DCt
        self.stft = ConvSTFT(**configs.stft_configs.dict())
        self.istft = ConvISTFT(**configs.stft_configs.dict())
        self.dct = LinearDCT(n_dct=self.stft.dim)
        self.cep_clean_encoder = (
            Encoder(**configs.cep_encoder_configs.dict()) 
            if configs.cep_encoder_configs is not None else None)
        self.cep_noisy_encoder = (
            Encoder(**configs.cep_encoder_configs.dict()) 
            if configs.cep_encoder_configs is not None else None)
        
        # Speech variance estimator
        self.d_vae = DenoisingVAEUNet(**configs.speech_variance_configs.dict())

        # Noise variance estimator
        self.e_noise = NoiseEstimator(**configs.noise_estimator_configs.dict())

        # Phase corrector
        self.phase_corrector = ComplexCRNN(**configs.phase_corrector_configs.dict())

        # Wiener type
        self.wiener_type: Literal['original', 'irm'] = configs.wiener_type

        # Training strategy
        self.latent_denoising_epochs: int = configs.latent_denoising_epochs
        self.noise_estimation_epochs: int = configs.noise_estimation_epochs

    @property
    def output_logging_keys(self) -> List[str]:
        return (
            ['loss', 'recon_xY', 'recon_n', 'condition_loss', 'sisdr_db_xHat1', 'sisdr_db_xHat2', 'sisdr_db_y']
            + [f'dkl_{_k}' for _k in range(len(self.d_vae.noisy_encoders.encoders))])

    def train(self: 'SpeechEnhancementVAE', mode: bool = True) -> 'SpeechEnhancementVAE':
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        self.stft.train(mode=mode)
        self.istft.train(mode=mode)
        if self.cep_noisy_encoder is not None:
            self.dct.train(mode=mode)
            self.cep_clean_encoder.train(mode=False)
            self.cep_noisy_encoder.train(mode=mode)
        self.d_vae.train(mode=mode)
        self.e_noise.train(mode=mode)
        self.phase_corrector.train(mode=mode)

        return self

    def load_state_dict_from_pretrain_state_dict(
            self,
            pretrain_net_dict: Dict[str, Any]):
        #* The current model (self) and the state dict must be 
        #* on CPU when running this function!
        pretrain_net_configs = PretrainSpeechVarianceVAE.ConstructorArgs(**pretrain_net_dict['configs'])
        pretrain_net_state_dict = pretrain_net_dict['state_dict']
        pretrain_net = PretrainSpeechVarianceVAE(**pretrain_net_configs.dict())
        pretrain_net.load_state_dict(pretrain_net_state_dict)

        self.d_vae.clean_encoders.load_state_dict(pretrain_net.vae.encoders.state_dict())
        self.d_vae.decoders.load_state_dict(pretrain_net.vae.decoders.state_dict())

        if self.cep_clean_encoder is not None:
            self.cep_clean_encoder.load_state_dict(pretrain_net.cep_encoder.state_dict())

    def forward(self, x_bt: torch.Tensor, y_bt: torch.Tensor, epoch: int = -1) -> Tuple[torch.Tensor, Dict[str, float]]:
        eps = 1e-12
        log_eps = -12 * math.log(10)
        with torch.no_grad():
            X_b2ft = self.stft(x_bt)
            Xp_bft = X_b2ft.square().sum(dim=1)
            logXp_bft = (Xp_bft + eps).log()

            Y_b2ft = self.stft(y_bt)
            Yp_bft = Y_b2ft.square().sum(dim=1)
            logYp_bft = (Yp_bft + eps).log()

            n_bt = y_bt - x_bt
            N_b2ft = self.stft(n_bt)
            Np_bft = N_b2ft.square().sum(dim=1)
            logNp_bft = (Np_bft + eps).log()
    
        list_cdx_bct = None if self.cep_clean_encoder is None else [
            None, self.cep_clean_encoder(
                self.dct(logXp_bft.transpose(-1, -2)).transpose(-1, -2))]
        list_cdy_bct = None if self.cep_noisy_encoder is None else [
            None, self.cep_noisy_encoder(
                self.dct(logYp_bft.transpose(-1, -2)).transpose(-1, -2))]
        logXpHatX_bft, list_msx_bct, logXpHatY_bft, list_msy_bct = self.d_vae(
            logXp_bft, logYp_bft,
            list_cdx_bct=list_cdx_bct,
            list_cdy_bct=list_cdy_bct)

        # The autoencoder encodes the input sequence with some additional 
        # future information and decodes the original input sequence.
        logXpHatX_bft = future_estimation(logXpHatX_bft, self.look_ahead_frames).clamp(log_eps, -log_eps)
        logXpHatY_bft = future_estimation(logXpHatY_bft, self.look_ahead_frames).clamp(log_eps, -log_eps)

        if epoch >= self.latent_denoising_epochs:
            logNpHat_bft: torch.Tensor = self.e_noise(
                logXpHatY_bft.detach(),
                past_groundtruth(logYp_bft, logXpHatY_bft))

            # Ideal ratio mask
            HpHat_bft = 1. / (1. + (logNpHat_bft - past_groundtruth(logXpHatY_bft, logNpHat_bft)).exp())
            if self.wiener_type == 'irm':
                XHat1_b2ft = HpHat_bft.detach().unsqueeze(dim=1).sqrt() * past_groundtruth(Y_b2ft, HpHat_bft)
            elif self.wiener_type == 'original':
                XHat1_b2ft = HpHat_bft.detach().unsqueeze(dim=1) * past_groundtruth(Y_b2ft, HpHat_bft)
            else:
                raise KeyError('Unexpected Wiener type! Must be either "irm" or "original"!')

            # Waveform reconstruction
            with torch.no_grad():
                x_bt = self.istft(X_b2ft)
                y_bt = self.istft(Y_b2ft)
                xHat1_bt = self.istft(XHat1_b2ft)

            if epoch >= self.latent_denoising_epochs + self.noise_estimation_epochs:
                # Phase correction
                MPhiHat_b2ft: torch.Tensor = self.phase_corrector(XHat1_b2ft)
                MPhiHat_b2ft = future_estimation(MPhiHat_b2ft, self.look_ahead_frames)

                XHat2_b2ft = complex_mask_multiplication(
                    past_groundtruth(XHat1_b2ft, MPhiHat_b2ft),
                    MPhiHat_b2ft)

                # Waveform reconstruction
                xHat2_bt = self.istft(XHat2_b2ft)

        # -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o- Loss computation -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-
        if epoch >= self.latent_denoising_epochs:
            # --- Speech reconstruction loss
            if self.spectral_distribution == 'exponential':
                recon_xY = kl_divergence_exponential(
                    past_groundtruth(logXp_bft, logXpHatY_bft), logXpHatY_bft)
            elif self.spectral_distribution == 'log-normal':
                recon_xY = kl_divergence_log_normal_unit_variance(
                    past_groundtruth(logXp_bft, logXpHatY_bft), logXpHatY_bft)
        else:
            recon_xY = torch.tensor(0.0).float().to(x_bt.device)

        # --- Condition loss
        condition_loss = torch.zeros_like(recon_xY) if self.cep_noisy_encoder is None or epoch >= self.latent_denoising_epochs else F.mse_loss(list_cdy_bct[-1], list_cdx_bct[-1])
        
        # --- Latent denoising: Kullback-Leibler divergence between two categorical distribution
        list_dkl = [
            kl_divergence_normal(
                muA__=muX_i_bct.detach(), logSig2A__=logSig2X_i_bct.detach(),
                muB__=muY_i_bct, logSig2B__=logSig2Y_i_bct,
                dim=1)
            for (muX_i_bct, logSig2X_i_bct), (muY_i_bct, logSig2Y_i_bct) in zip(
                list_msx_bct, list_msy_bct)]

        # --- Noise reconstruction error
        if epoch >= self.latent_denoising_epochs:
            if self.spectral_distribution == 'exponential':
                recon_n = kl_divergence_exponential(
                    past_groundtruth(logNp_bft, logNpHat_bft), logNpHat_bft)
            elif self.spectral_distribution == 'log-normal':
                recon_n = kl_divergence_log_normal_unit_variance(
                    past_groundtruth(logNp_bft, logNpHat_bft), logNpHat_bft)
        else:
            recon_n = torch.zeros_like(recon_xY)
        
        # --- Negative signal-to-distortion ratio loss for phase reconstruction
        if epoch >= self.latent_denoising_epochs:
            with torch.no_grad():
                nsisdr_db_y = negative_signal_to_distortion_ratio_decibel(x_bt, y_bt)
                nsisdr_db_xHat1 = negative_signal_to_distortion_ratio_decibel(
                    past_groundtruth(x_bt, xHat1_bt), xHat1_bt)
        else:
            nsisdr_db_y = torch.zeros_like(recon_xY)
            nsisdr_db_xHat1 = torch.zeros_like(recon_xY)

        if epoch >= self.latent_denoising_epochs + self.noise_estimation_epochs:
            nsisdr_db_xHat2 = negative_signal_to_distortion_ratio_decibel(
                past_groundtruth(x_bt, xHat2_bt), xHat2_bt)
        else:
            nsisdr_db_xHat2 = torch.zeros_like(recon_xY)

        # Total loss
        loss = (
            recon_xY + condition_loss
            + sum(list_dkl)
            + recon_n
            + nsisdr_db_xHat2)

        return loss, {
            'loss': loss.item(),
            'recon_xY': recon_xY.item(),
            'recon_n': recon_n.item(),
            'condition_loss': condition_loss.item(),
            'sisdr_db_xHat1': -nsisdr_db_xHat1.item(),
            'sisdr_db_xHat2': -nsisdr_db_xHat2.item(),
            'sisdr_db_y': -nsisdr_db_y.item(),
            **{
                f'dkl_{_k}': cl.item() 
                for _k, cl in enumerate(list_dkl)}}

    def validate(self, x_bt: torch.Tensor, y_bt: torch.Tensor, epoch: int = -1) -> Dict[str, Any]:
        eps = 1e-12
        log_eps = -12 * math.log(10)
        pad_len = compute_pad_len_for_inference(
            x_bt, stft_win_len=self.stft.win_len, stft_hop_len=self.stft.hop_len,
            required_stride=self.d_vae.clean_encoders.total_stride,
            look_ahead_frames=self.look_ahead_frames)

        with torch.no_grad():
            x_bt = F.pad(x_bt, (0, pad_len))
            X_b2ft = self.stft(x_bt)
            Xp_bft = X_b2ft.square().sum(dim=1)
            logXp_bft = (Xp_bft + eps).log()

            y_bt = F.pad(y_bt, (0, pad_len))
            Y_b2ft = self.stft(y_bt)
            Yp_bft = Y_b2ft.square().sum(dim=1)
            logYp_bft = (Yp_bft + eps).log()

            n_bt = y_bt - x_bt
            N_b2ft = self.stft(n_bt)
            Np_bft = N_b2ft.square().sum(dim=1)
            logNp_bft = (Np_bft + eps).log()

            list_cdx_bct = None if self.cep_clean_encoder is None else [
                None, self.cep_clean_encoder(
                    self.dct(logXp_bft.transpose(-1, -2)).transpose(-1, -2))]
            list_cdy_bct = None if self.cep_noisy_encoder is None else [
                None, self.cep_noisy_encoder(
                    self.dct(logYp_bft.transpose(-1, -2)).transpose(-1, -2))]
            logXpHatX_bft, list_msx_bct, logXpHatY_bft, list_msy_bct = self.d_vae(
                logXp_bft, logYp_bft,
                list_cdx_bct=list_cdx_bct,
                list_cdy_bct=list_cdy_bct)

            # The autoencoder encodes the input sequence with some additional 
            # future information and decodes the original input sequence.
            logXpHatX_bft = future_estimation(logXpHatX_bft, self.look_ahead_frames).clamp(log_eps, -log_eps)
            logXpHatY_bft = future_estimation(logXpHatY_bft, self.look_ahead_frames).clamp(log_eps, -log_eps)

            if epoch >= self.latent_denoising_epochs:
                logNpHat_bft: torch.Tensor = self.e_noise(
                    logXpHatY_bft.detach(),
                    past_groundtruth(logYp_bft, logXpHatY_bft))

                # Ideal ratio mask
                HpHat_bft = 1. / (1. + (logNpHat_bft - past_groundtruth(logXpHatY_bft, logNpHat_bft)).exp())
                if self.wiener_type == 'irm':
                    XHat1_b2ft = HpHat_bft.detach().unsqueeze(dim=1).sqrt() * past_groundtruth(Y_b2ft, HpHat_bft)
                elif self.wiener_type == 'original':
                    XHat1_b2ft = HpHat_bft.detach().unsqueeze(dim=1) * past_groundtruth(Y_b2ft, HpHat_bft)
                else:
                    raise KeyError('Unexpected Wiener type! Must be either "irm" or "original"!')

                # Waveform reconstruction
                with torch.no_grad():
                    x_bt = self.istft(X_b2ft)
                    y_bt = self.istft(Y_b2ft)
                    xHat1_bt = self.istft(XHat1_b2ft)

                if epoch >= self.latent_denoising_epochs + self.noise_estimation_epochs:
                    # Phase correction
                    MPhiHat_b2ft: torch.Tensor = self.phase_corrector(XHat1_b2ft)
                    MPhiHat_b2ft = future_estimation(MPhiHat_b2ft, self.look_ahead_frames)

                    XHat2_b2ft = complex_mask_multiplication(
                        past_groundtruth(XHat1_b2ft, MPhiHat_b2ft),
                        MPhiHat_b2ft)

                    # Waveform reconstruction
                    xHat2_bt = self.istft(XHat2_b2ft)

            # -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o- Loss computation -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-
            # --- Speech reconstruction loss
            if epoch >= self.latent_denoising_epochs:
                if self.spectral_distribution == 'exponential':
                    recon_xY = kl_divergence_exponential(
                        past_groundtruth(logXp_bft, logXpHatY_bft), logXpHatY_bft)
                elif self.spectral_distribution == 'log-normal':
                    recon_xY = kl_divergence_log_normal_unit_variance(
                        past_groundtruth(logXp_bft, logXpHatY_bft), logXpHatY_bft)
            else:
                recon_xY = torch.tensor(0.0).float().to(x_bt.device)
            
            # --- Condition loss
            condition_loss = torch.zeros_like(recon_xY) if self.cep_noisy_encoder is None or epoch >= self.latent_denoising_epochs else F.mse_loss(list_cdy_bct[-1], list_cdx_bct[-1])

            # --- Latent denoising: Kullback-Leibler divergence between two categorical distribution
            list_dkl = [
                kl_divergence_normal(
                    muA__=muX_i_bct.detach(), logSig2A__=logSig2X_i_bct.detach(),
                    muB__=muY_i_bct, logSig2B__=logSig2Y_i_bct,
                    dim=1)
                for (muX_i_bct, logSig2X_i_bct), (muY_i_bct, logSig2Y_i_bct) in zip(
                    list_msx_bct, list_msy_bct)]

            # --- Noise reconstruction error
            if epoch >= self.latent_denoising_epochs:
                if self.spectral_distribution == 'exponential':
                    recon_n = kl_divergence_exponential(
                        past_groundtruth(logNp_bft, logNpHat_bft), logNpHat_bft)
                elif self.spectral_distribution == 'log-normal':
                    recon_n = kl_divergence_log_normal_unit_variance(
                        past_groundtruth(logNp_bft, logNpHat_bft), logNpHat_bft)
            else:
                recon_n = torch.zeros_like(recon_xY)
            
            # --- Negative signal-to-distortion ratio loss for phase reconstruction
            if epoch >= self.latent_denoising_epochs:
                with torch.no_grad():
                    nsisdr_db_y = negative_signal_to_distortion_ratio_decibel(x_bt, y_bt)
                    nsisdr_db_xHat1 = negative_signal_to_distortion_ratio_decibel(
                        past_groundtruth(x_bt, xHat1_bt), xHat1_bt)
            else:
                nsisdr_db_y = torch.zeros_like(recon_xY)
                nsisdr_db_xHat1 = torch.zeros_like(recon_xY)

            if epoch >= self.latent_denoising_epochs + self.noise_estimation_epochs:
                nsisdr_db_xHat2 = negative_signal_to_distortion_ratio_decibel(
                    past_groundtruth(x_bt, xHat2_bt), xHat2_bt)
            else:
                nsisdr_db_xHat2 = torch.zeros_like(recon_xY)

            # Total loss
            loss = (
                recon_xY + condition_loss
                + sum(list_dkl)
                + recon_n
                + nsisdr_db_xHat2)
            return {
                'numerical': {
                    'loss': loss.item(),
                    'recon_xY': recon_xY.item(),
                    'recon_n': recon_n.item(),
                    'condition_loss': condition_loss.item(),
                    'sisdr_db_xHat1': -nsisdr_db_xHat1.item(),
                    'sisdr_db_xHat2': -nsisdr_db_xHat2.item(),
                    'sisdr_db_y': -nsisdr_db_y.item(),
                    **{
                        f'dkl_{_k}': cl.item() 
                        for _k, cl in enumerate(list_dkl)}},
                'waveform': {
                    'x_bt': x_bt.detach().cpu(),
                    'y_bt': y_bt.detach().cpu(),
                    'xHat1_bt': xHat1_bt.detach().cpu(),
                    'xHat2_bt': xHat2_bt.detach().cpu()},
                'spectrum': {
                    'logXp_bft': logXp_bft.detach().cpu(),
                    'logYp_bft': logYp_bft.detach().cpu(),
                    'logNp_bft': logNp_bft.detach().cpu(),
                    'logXpHatX_bft': logXpHatX_bft.detach().cpu(),
                    'logXpHatY_bft': logXpHatY_bft.detach().cpu() if logXpHatY_bft is not None else torch.zeros_like(logXpHatX_bft).detach_().cpu(),
                    'logNpHat_bft': logNpHat_bft.detach().cpu() if logNpHat_bft is not None else torch.zeros_like(logXpHatX_bft).detach_().cpu()},
                'mask': {
                    'HpHat_bft': HpHat_bft.detach().cpu()}}

    def enhance(self, y_bt: torch.Tensor, phase_correction: bool = False) -> torch.Tensor:
        eps = 1e-12
        log_eps = -12 * math.log(10)

        with torch.no_grad():
            # Calculate spectrogram
            Y_b2ft: torch.Tensor = self.stft(y_bt)

            available_len = Y_b2ft.size(-1) - (Y_b2ft.size(-1) % self.d_vae.clean_encoders.total_stride)
            Y_b2ft = Y_b2ft[..., : available_len]

            # Log noisy spectrogram
            logYp_bft: torch.Tensor = (Y_b2ft.square().sum(dim=1) + eps).log()

            # Estimated log variance of speech
            list_cdy_bct = None if self.cep_noisy_encoder is None else [
                None, self.cep_noisy_encoder(
                    self.dct(logYp_bft.transpose(-1, -2)).transpose(-1, -2))]
            logXpHatY_bft: torch.Tensor = self.d_vae.denoise(logYp_bft, list_cdy_bct=list_cdy_bct)
            logXpHatY_bft = future_estimation(logXpHatY_bft, self.look_ahead_frames).clamp(log_eps, -log_eps)

            # Noise variance estimation
            logNpHat_bft: torch.Tensor = self.e_noise(
                logXpHatY_bft, 
                past_groundtruth(logYp_bft, logXpHatY_bft))
            logNpHat_bft = future_estimation(logNpHat_bft, self.look_ahead_frames).clamp(log_eps, -log_eps)

            # Ideal ratio mask from Wiener filter
            HpHat_bft = 1. / (1. + (logNpHat_bft - past_groundtruth(logXpHatY_bft, logNpHat_bft)).exp())

            if self.wiener_type == 'irm':
                XHat_b2ft = HpHat_bft.unsqueeze(dim=1).sqrt() * past_groundtruth(Y_b2ft, HpHat_bft)
            elif self.wiener_type == 'original':
                XHat_b2ft = HpHat_bft.unsqueeze(dim=1) * past_groundtruth(Y_b2ft, HpHat_bft)
            else:
                raise KeyError('Unexpected Wiener type! Must be either "irm" or "original"!')

            # Phase correction
            if phase_correction:
                MPhiHat_b2ft: torch.Tensor = self.phase_corrector(XHat_b2ft)
                MPhiHat_b2ft = future_estimation(MPhiHat_b2ft, self.look_ahead_frames)

                XHat_b2ft = complex_mask_multiplication(
                    past_groundtruth(XHat_b2ft, MPhiHat_b2ft),
                    MPhiHat_b2ft)

            xHat_bt: torch.Tensor = self.istft(XHat_b2ft)

            return xHat_bt
