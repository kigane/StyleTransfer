import torch
import torch.nn as nn


def get_truncated_noise(n_samples, z_dim, truncation):
    '''创建维度为(n_samples, z_dim)的服从截断正态分布的噪声向量。truncation越小则生成图像的真实性越好，多样性越差，反之则多样性更好，真实性较差。

    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        truncation: the truncation value, a non-negative scalar
    '''
    truncated_noise = torch.empty(n_samples, z_dim)
    nn.init.trunc_normal_(truncated_noise, a=-1*truncation, b=truncation)
    return truncated_noise
