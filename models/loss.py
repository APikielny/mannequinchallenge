import torch

def L2(latent_1, latent_2):
    return torch.mean((latent_1 - latent_2) ** 2)