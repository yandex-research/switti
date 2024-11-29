import torch


def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def compute_axial_cis(
    dim: int, end_x: int, end_y: int, theta: float = 100.0, norm_coeff: int = 1
):
    freqs_x = (
        1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
        * norm_coeff
    )
    freqs_y = (
        1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
        * norm_coeff
    )

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    freqs_cis = freqs_cis[:, x.shape[1], ...]
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor):
    with torch.cuda.amp.autocast(enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        # freqs_cis = reshape_for_broadcast(freqs_cis, x).to(x_in.device)
        freqs_cis = freqs_cis[None, :, : x.shape[2], ...].to(x_in.device)
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)
        return x_out.type_as(x_in)
