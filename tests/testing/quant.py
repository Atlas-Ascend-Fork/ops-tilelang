import torch

from cann_ops_tilelang.quant.common import get_packed_ue8m0_pack_factor, unpack_from_e2m1fn_x2
from cann_ops_tilelang.utils import align_up, ceil_div


def quant_level_rank(xc: torch.Tensor, fmt: str) -> torch.Tensor:
    """Map each fp8/fp4 code to its rank in the value-sorted list of representable levels.

    Two outputs are 'numerically adjacent' iff their ranks differ by at most 1, which holds
    across zero and exponent boundaries (unlike raw bit-pattern subtraction).
    """
    if fmt == 'e4m3':
        codes = torch.arange(256, dtype=torch.uint8, device=xc.device)
        values = codes.view(torch.float8_e4m3fn).float()
        q = xc.float()
    else:
        codes = torch.arange(16, dtype=torch.uint8, device=xc.device)
        values = unpack_from_e2m1fn_x2(codes.view(-1, 1)).flatten()
        q = unpack_from_e2m1fn_x2(xc).float()
    finite = torch.isfinite(values)
    levels = torch.unique(values[finite])  # sorted ascending
    return torch.searchsorted(levels, q.flatten())


def clear_unused_sf(sf: torch.Tensor, hidden: int, num_per_channels: int) -> torch.Tensor:
    """Zero out unused sf entries beyond the actual channel block count.

    Args:
        sf: Scale-factor tensor to clean up.
        hidden: Number of hidden channels in the original tensor.
        num_per_channels: Number of channels per scaling block.

    Returns:
        Flattened sf tensor with unused trailing entries set to zero.
    """
    num_channel_blocks = ceil_div(hidden, num_per_channels)
    aligned_num_channel_blocks = align_up(num_channel_blocks, get_packed_ue8m0_pack_factor())
    sf_flattened = sf.contiguous().flatten()
    sf_flattened = (sf_flattened.as_strided(sf_flattened.shape, (1,)).view(torch.uint8)).view(
        -1, aligned_num_channel_blocks
    )
    sf_flattened[:, num_channel_blocks:] = 0
    return sf_flattened


def clear_unused_sf_col_pack(sf: torch.Tensor, num_tokens: int, num_per_tokens: int) -> torch.Tensor:
    """Zero out unused sf entries beyond the actual token block count.

    Args:
        sf: Scale-factor tensor to clean up.
        num_tokens: Number of tokens in the original tensor.
        num_per_tokens: Number of tokens per scaling block.

    Returns:
        uint8 view of sf with unused trailing entries set to zero.
    """
    num_token_blocks = ceil_div(num_tokens, num_per_tokens)
    pack_factor = get_packed_ue8m0_pack_factor()
    aligned_num_token_blocks = align_up(num_token_blocks, pack_factor)
    if sf.numel() == 0:
        return sf.view(torch.uint8)
    sf_flattened = sf.contiguous().flatten()
    sf_uint8 = sf_flattened.as_strided(sf_flattened.shape, (1,)).view(torch.uint8)
    sf_unpacked = sf_uint8.view(aligned_num_token_blocks // pack_factor, -1, pack_factor)
    sf_unpacked[num_token_blocks // pack_factor :, :, num_token_blocks % pack_factor :] = 0
    return sf_uint8.view(aligned_num_token_blocks // pack_factor, -1)
