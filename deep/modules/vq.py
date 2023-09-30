import math
import random
import torch
from pydantic import BaseModel
from torch import nn, distributions
from torch.nn import functional as F
from typing import Dict, Any, List, Optional, Tuple


class ReservoirSampler(nn.Module):
    def __init__(self, n_reservoir_samples: int, embedding_dim: int, dtype: Optional[torch.dtype] = None) -> None:
        super(ReservoirSampler, self).__init__()
        self.n_reservoir_samples: int = n_reservoir_samples
        self.embedding_dim: int = embedding_dim

        # Number of observed samples
        self.register_buffer('n_observed_samples', torch.tensor(0, dtype=torch.int64))

        # The most recent sample index to be chosen to the reservoir
        self.register_buffer('current_index', torch.tensor(0, dtype=torch.int64))

        # For algorithm L
        self.register_buffer('w_gen', torch.tensor(1.0, dtype=torch.float))

        # Reservoir samples
        self.register_buffer('r_ld', torch.empty((self.n_reservoir_samples, embedding_dim), dtype=dtype))

    def reset_reservoir(self) -> None:
        self.n_observed_samples.fill_(0)
        self.current_index.fill_(0)
        self.w_gen.fill_(1.0)

    @staticmethod
    def u() -> float:
        """Random a float in (0, 1)

        Returns:
            float: Output number
        """
        eps = 1e-6
        return min(max(random.random(), eps), 1.0 - eps)

    @property
    def collect_enough(self) -> bool:
        return self.current_index.item() >= self.n_reservoir_samples

    def update_from_data(self, x_nd: torch.Tensor) -> None:
        with torch.no_grad():
            n = x_nd.size(0)
            previous_n_observed_samples = self.n_observed_samples.item()
            self.n_observed_samples.add_(n)

            # If the current number of reservoir samples are not enough
            if self.current_index.item() < self.n_reservoir_samples:
                cur_idx = self.current_index.item()
                assert cur_idx == previous_n_observed_samples

                # Use as much observed samples as possible for reservoir
                Q = min(self.n_observed_samples.item(), self.n_reservoir_samples) - cur_idx

                self.r_ld[cur_idx: cur_idx + Q, :].copy_(x_nd[: Q, ...])
                self.current_index.add_(Q)

            # Implementation trick: Force to choose the immediate next sample to the reservoir
            if self.current_index.item() == self.n_reservoir_samples:
                self.current_index.add_(1)

            # If the number of reservoir samples are enough already
            if self.current_index.item() > self.n_reservoir_samples:
                # Use dict to keep the latest replacements only
                candidate_dict = dict()

                while self.current_index.item() <= self.n_observed_samples.item():
                    candidate_idx = self.current_index.item() - previous_n_observed_samples - 1
                    updated_idx = random.randrange(self.n_reservoir_samples)
                    candidate_dict[updated_idx] = candidate_idx

                    self.w_gen.mul_(math.exp(math.log(self.u()) / self.n_reservoir_samples))
                    self.w_gen.clamp_(min=1e-6, max=1.0 - 1e-6)
                    self.current_index.add_(math.floor(math.log(self.u()) / math.log(1.0 - self.w_gen.item())) + 1)

                if len(candidate_dict) > 0:
                    candidate_indices = []
                    updated_indices = []
                    for updated_idx, candidate_idx in candidate_dict.items():
                        candidate_indices.append(candidate_idx)
                        updated_indices.append(updated_idx)

                    i_n = torch.tensor(data=candidate_indices, device=x_nd.device)
                    l_n = torch.tensor(data=updated_indices, device=x_nd.device)

                    self.r_ld.index_copy_(dim=0, index=l_n, source=x_nd.index_select(dim=0, index=i_n).float())

    def get_reservoir_samples(self) -> torch.Tensor:
        return self.r_ld.clone()


class Codebook(nn.Embedding):
    class ConstructorArgs(BaseModel):
        dim_codebook: int
        codebook_size: int

    def __init__(
            self,
            dim_codebook: int, codebook_size: int, **kwargs) -> None:
        super(Codebook, self).__init__(
            num_embeddings=codebook_size,
            embedding_dim=dim_codebook,
            **kwargs)
        self.weight.data.uniform_(-1 / codebook_size, 1 / codebook_size)
        self._eps: float = torch.finfo(torch.float32).eps

    @property
    def dim_codebook(self) -> int:
        return self.embedding_dim

    @property
    def codebook_size(self) -> int:
        return self.num_embeddings

    @staticmethod
    def pairwise_distance(
            u_nd: torch.Tensor, v_md: torch.Tensor) -> torch.Tensor:
        v_dm = v_md.t()
        d_nm = torch.addmm(
            input=(u_nd.square().sum(-1, keepdim=True) 
                   + v_dm.square().sum(0, keepdim=True)),
            mat1=u_nd, mat2=v_dm, beta=1, alpha=-2)
        
        return d_nm

    @staticmethod
    def perplexity(oneHot__k: torch.Tensor) -> torch.Tensor:
        oneHot_nk = oneHot__k.flatten(start_dim=0, end_dim=-2)
        pSelected_k = oneHot_nk.float().mean(dim=0)
        entropy = -torch.sum(pSelected_k * torch.log(pSelected_k + 1e-10))
        results = torch.exp(entropy)
        return results

    def lookup(self, idx__: torch.Tensor) -> torch.Tensor:
        return self(idx__)

    def quantise(
            self, x__d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x__d (torch.Tensor): Input vectors
        Return:
            p__k (torch.Tensor): Estimated categorical posterior distributions
            oneHot__k (torch.Tensor): Samples from the estimated posteriors
            d__k (torch.Tensor): Distance to each embedding in the codebook
            xq__d (torch.Tensor): Corresponding vectors to the samples
        """
        input_size = x__d.size()[: -1]  # Except last dimension
        x_nd = x__d.flatten(start_dim=0, end_dim=-2)
        
        d_nk = self.pairwise_distance(x_nd, self.weight)
        idx_n = torch.argmin(d_nk, dim=-1)
        xq_nd = self.lookup(idx_n)
        oneHot_nk = F.one_hot(idx_n, num_classes=self.codebook_size).float()

        d__k = d_nk.unflatten(dim=0, sizes=input_size)
        xq__d = xq_nd.unflatten(dim=0, sizes=input_size)
        oneHot__k = oneHot_nk.unflatten(dim=0, sizes=input_size)
        p__k = oneHot__k.clone()

        return p__k, oneHot__k, d__k, xq__d

    def quantise_with_logs(
            self, x__d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        p__k, oneHot__k, d__k, xq__d = self.quantise(x__d)

        return p__k, oneHot__k, d__k, xq__d, {
            'perplexity': self.perplexity(oneHot__k)
        }


class EMACodebook(Codebook):
    class ConstructorArgs(BaseModel):
        dim_codebook: int
        codebook_size: int
        gamma: float = 0.99
        epsilon: float = 1e-5
        n_reservoir_samples: Optional[int] = None

    def __init__(
            self, 
            dim_codebook: int, 
            codebook_size: int,
            gamma: float = 0.99,
            epsilon: float = 1e-5,
            n_reservoir_samples: Optional[int] = None,
            **kwargs) -> None:
        super().__init__(dim_codebook, codebook_size, **kwargs)
        self.gamma: float = gamma
        self.epsilon: float = epsilon

        # EMA cluster size
        self.register_buffer('N_k', torch.ones(codebook_size))
        # EMA running sum
        self.register_buffer('m_kd', self.weight.data.clone().requires_grad_(False))

        # Reservoir sampling
        if n_reservoir_samples is not None:
            assert n_reservoir_samples > codebook_size, \
                'Number of reservoir samples must be larger than codebook size!'

            self.reservoir_sampler = ReservoirSampler(
                n_reservoir_samples=n_reservoir_samples,
                embedding_dim=dim_codebook,
                dtype=self.weight.dtype)
        else:
            self.reservoir_sampler = None

    def set_codebook_ema_momentum(self, lr: Optional[float] = None) -> None:
        """Update momentum of EMA method from model's learning rate

        Args:
            lr (Optional[float], optional): Model learning rate. Defaults to None.
        """
        # if lr is not None:
        #     # Codebook learning rate is set to 10 times larger than model learning rate
        #     codebook_lr = 100. * lr   

        #     if codebook_lr < 0.5:
        #         self.gamma = 1. - 2. * codebook_lr
        pass

    def update_reservoir(self, x__d: torch.Tensor) -> None:
        if self.reservoir_sampler is not None and self.training:
            with torch.no_grad():
                x_nd = x__d.flatten(start_dim=0, end_dim=-2).detach()
                self.reservoir_sampler.update_from_data(x_nd=x_nd)
               
    def initialise_codebook_from_reservoir(self) -> None:
        """ K-Mean++ algorithm from reservoir samples """
        if self.reservoir_sampler is not None and self.training:
            # Must collect enough reservoir samples
            assert self.reservoir_sampler.collect_enough, 'Reservoir sampler must collect enough samples!'
            with torch.no_grad():
                L = self.reservoir_sampler.n_reservoir_samples  # Number of unused reservoir samples
                K = self.codebook_size
                r_ld = self.reservoir_sampler.get_reservoir_samples()

                avail_reservoir_samples: List[int] = list(range(L))
                selected_centre_samples: List[int] = []
                for k in range(K):
                    if k == 0:
                        idx = random.randrange(L)
                    else:
                        # Pairwise distance between current centres and unused reservoir samples
                        iR_l = torch.tensor(data=avail_reservoir_samples, dtype=torch.int, device=self.weight.device)
                        iC_k = torch.tensor(data=selected_centre_samples, dtype=torch.int, device=self.weight.device)
                        d_l, _ = self.pairwise_distance(
                            r_ld.index_select(dim=0, index=iR_l),
                            r_ld.index_select(dim=0, index=iC_k)
                        ).min(dim=-1)

                        # Probability (weight) of selecting is proportional to the (squared) distance
                        idx = random.choices(population=range(L), weights=d_l.tolist(), k=1)[0]

                    selected_centre_samples.append(avail_reservoir_samples.pop(idx))
                    L -= 1

                iC_k = torch.tensor(data=selected_centre_samples, dtype=torch.int, device=self.weight.device)
                self.weight.data.copy_(r_ld.index_select(dim=0, index=iC_k))

                self.N_k.fill_(1.0)
                self.m_kd.copy_(self.weight.data)

                self.reservoir_sampler.reset_reservoir()

    def quantise(
            self, x__d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x__d (torch.Tensor): Input vectors
        Return:
            p__k (torch.Tensor): Estimated categorical posterior distributions
            oneHot__k (torch.Tensor): Samples from the estimated posteriors
            d__k (torch.Tensor): Distance to each embedding in the codebook
            xq__d (torch.Tensor): Corresponding vectors to the samples
        """
        input_size = x__d.size()[: -1]  # Except last dimension
        x_nd = x__d.flatten(start_dim=0, end_dim=-2)
        
        d_nk = self.pairwise_distance(x_nd, self.weight.detach())
        idx_n = torch.argmin(d_nk, dim=-1)
        oneHot_nk = F.one_hot(idx_n, num_classes=self.codebook_size).float()

        if self.training:
            with torch.no_grad():
                # Update cluster size using EMA
                n_k = oneHot_nk.sum(dim=0)
                self.N_k.mul_(self.gamma).add_((1 - self.gamma) * n_k)
                
                # Laplace smoothing to avoid empty clusters
                N_1 = self.N_k.sum()
                self.N_k.add_(self.epsilon).div_(N_1 + self.codebook_size * self.epsilon).mul_(N_1)

                # Update running sum
                dm_kd = oneHot_nk.t() @ x_nd
                self.m_kd.mul_(self.gamma).add_((1 - self.gamma) * dm_kd)

            # Update codebook
            self.weight.data.copy_(self.m_kd / self.N_k.unsqueeze(-1))

        # The selected codes are from updated codebook (the centroids of the batch)
        xq_nd = self.lookup(idx_n)
        xq__d = xq_nd.unflatten(dim=0, sizes=input_size)

        d__k = d_nk.unflatten(dim=0, sizes=input_size)
        oneHot__k = oneHot_nk.unflatten(dim=0, sizes=input_size)
        p__k = oneHot__k.clone()

        return p__k, oneHot__k, d__k, xq__d


class GumbelCodebook(Codebook):
    class ConstructorArgs(BaseModel):
        dim_codebook: int
        codebook_size: int
        tau: float = 0.5

    def __init__(
            self, 
            dim_codebook: int, 
            codebook_size: int, 
            tau: float = 0.5,
            **kwargs) -> None:
        super().__init__(dim_codebook, codebook_size, **kwargs)
        self.tau: float = tau

    def quantise(
            self, x__d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x__d (torch.Tensor): Input vectors
        Return:
            p__k (torch.Tensor): Estimated categorical posterior distributions
            oneHot__k (torch.Tensor): Samples from the estimated posteriors
            d__k (torch.Tensor): Distance to each embedding in the codebook
            xq__d (torch.Tensor): Corresponding vectors to the samples
        """
        input_size = x__d.size()[: -1]  # Except last dimension
        x_nd = x__d.flatten(start_dim=0, end_dim=-2)

        d_nk = self.pairwise_distance(x_nd, self.weight)

        gumbel_softmax = distributions.RelaxedOneHotCategorical(
            logits=-d_nk, temperature=self.tau)
        p_nk = gumbel_softmax.probs
        oneHotSoft_nk = gumbel_softmax.rsample()

        if self.training:
            xq_nd = oneHotSoft_nk @ self.weight
        else:
            xq_nd = self.lookup(torch.argmax(oneHotSoft_nk, dim=-1))

        xq__d = xq_nd.unflatten(dim=0, sizes=input_size)
        oneHot__k = oneHotSoft_nk.unflatten(dim=0, sizes=input_size)
        p__k = p_nk.unflatten(dim=0, sizes=input_size)
        d__k = d_nk.unflatten(dim=0, sizes=input_size)

        return p__k, oneHot__k, d__k, xq__d


def sanity_check():
    import numpy as np
    from torch import optim
    from torch.nn import functional as F

    np.random.seed(0)
    torch.manual_seed(0)

    train_data_nd = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]], dtype=np.float32)
    test_data_nd = np.array([[0, 0], [12, 3]], dtype=np.float32)

    vq = Codebook(dim_codebook=2, codebook_size=2)
    vq_ema = EMACodebook(dim_codebook=2, codebook_size=2, gamma=0.5)
    vq_gumbel = GumbelCodebook(dim_codebook=2, codebook_size=2, tau=0.1)

    vq_optim = optim.Adam(vq.parameters(), lr=1.0)
    vq_gumbel_optim = optim.Adam(vq_gumbel.parameters(), lr=1.0)

    vq.train()
    vq_ema.train()
    vq_gumbel.train()
    for epoch in range(10):
        x_nd = torch.from_numpy(train_data_nd)

        # Train VQ
        vq_optim.zero_grad()
        _, oneHot_nk, _, xQ_nd = vq.quantise(x_nd.clone())
        perp = vq.perplexity(oneHot_nk)
        vq_loss = F.mse_loss(xQ_nd, x_nd.detach())
        vq_loss.backward()
        vq_optim.step()
        vq_loss_value = vq_loss.item()
        perp_value = perp.item()

        # Train VQ-EMA
        _, oneHot_nk, _, xQ_nd = vq_ema.quantise(x_nd.clone())
        perp = vq_ema.perplexity(oneHot_nk)
        vq_loss = F.mse_loss(xQ_nd, x_nd.detach())
        vqema_loss_value = vq_loss.item()
        perp_ema_value = perp.item()

        # Train VQ-Gumbel
        vq_gumbel_optim.zero_grad()
        p_nk, oneHot_nk, _, _ = vq_gumbel.quantise(x_nd.clone())
        perp = vq_gumbel.perplexity(oneHot_nk)
        dirac_nk = F.one_hot(p_nk.argmax(dim=-1), num_classes=2).float()
        vq_loss = (-(dirac_nk * (p_nk + 1e-12).log()).sum(dim=-1)).mean()  # Maximise log prob.
        vq_loss.backward()
        vq_gumbel_optim.step()
        vqgumbel_loss_value = vq_loss.item()
        perp_gumbel_value = perp.item()

        print(
            'Epoch {:d}: '
            'VQ_loss = {:.4f}, VQ_Perp = {:.4f}, '
            'VQEMA_loss = {:.4f}, VQEMA_Perp = {:.4f}, '
            'VQGumbel_loss = {:.4f}, VQGumbel_Perp = {:.4f}'.format(
                epoch, 
                vq_loss_value, perp_value, 
                vqema_loss_value, perp_ema_value,
                vqgumbel_loss_value, perp_gumbel_value))

    vq.eval()
    vq_ema.eval()
    vq_gumbel.eval()
    with torch.no_grad():
        x_nd = torch.from_numpy(test_data_nd)

        # Eval VQ
        _, oneHot_nk, _, _ = vq.quantise(x_nd.clone())
        idx_n = oneHot_nk.argmax(dim=-1)
        print(idx_n)
        print(vq.weight.data)

        # Eval VQ-EMA
        _, oneHot_nk, _, _ = vq_ema.quantise(x_nd.clone())
        idx_n = oneHot_nk.argmax(dim=-1)
        print(idx_n)
        print(vq_ema.weight.data)

        # Eval VQ-Gumbel
        _, oneHot_nk, _, _ = vq_gumbel.quantise(x_nd.clone())
        idx_n = oneHot_nk.argmax(dim=-1)
        print(idx_n)
        print(vq_gumbel.weight.data)


if __name__ == '__main__':
    sanity_check()
