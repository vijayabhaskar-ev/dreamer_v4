import torch
import math


def add_noise(z_clean, tau):
    tau = tau[:, :, None, None]
    z_noise = torch.randn_like(z_clean)
    z_noised = (1-tau)*z_noise + tau*z_clean
    return z_noised, z_noise


def sample_tau_and_d(batch_size, T,  K_max=64, device='cuda'):
    log2_K_max = int(math.log2(K_max))
    k_exp = torch.randint(0, log2_K_max + 1, (batch_size, T), device=device)
    k = 2**k_exp
    d = 1.0 / k.float() 

    tau_idx = torch.randint(0, K_max, (batch_size, T), device=device) % k
    tau = tau_idx.float() * d 
    return tau, d

    