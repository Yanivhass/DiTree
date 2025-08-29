import torch


def get_timesteps(schedule: str, k_steps: int, exp_scale: float = 1.0):
    t = torch.linspace(0, 1, k_steps + 1)[:-1]
    if schedule == "linear":
        dt = torch.ones(k_steps) / k_steps
    elif schedule == "cosine":
        dt = torch.cos(t * torch.pi) + 1
        dt /= torch.sum(dt)
    elif schedule == "exp":
        dt = torch.exp(-t * exp_scale)
        dt /= torch.sum(dt)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    t0 = torch.cat((torch.zeros(1), torch.cumsum(dt, dim=0)[:-1]))
    return t0, dt


# def sample_snr(batch_size: int, device) -> torch.Tensor:
#     if self.snr_sampler == "uniform":
#         return torch.rand((batch_size, 1, 1), device=device)
#     elif self.snr_sampler == "logit_normal":
#         return torch.sigmoid(torch.randn((batch_size, 1, 1), device=device))
#     else:
#         raise NotImplementedError
