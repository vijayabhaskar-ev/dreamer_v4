import torch
import math


def add_noise(z_clean, tau):
    tau = tau[:, :, None, None]
    z_noise = torch.randn_like(z_clean)
    z_noised = (1-tau)*z_noise + tau*z_clean
    return z_noised, z_noise


def sample_tau_and_d(batch_size, T,  K_max=64, device=None):
    log2_K_max = int(math.log2(K_max))

    # All RNG on-device via float ops — avoids CPU→TPU sync and int64 (TPU limitation).
    # Equivalent to: k_exp ~ Uniform{0,...,log2_K_max}, k = 2^k_exp
    k_exp = torch.rand(batch_size, T, device=device).mul(log2_K_max + 1).floor().clamp(max=log2_K_max)
    k = 2.0 ** k_exp          # float: 1.0, 2.0, 4.0, ..., K_max
    d = 1.0 / k

    # tau ~ Uniform{0, 1/k, 2/k, ..., (k-1)/k}
    # = floor(rand * k) / k
    tau = (torch.rand(batch_size, T, device=device) * k).floor() / k

    return tau, d
