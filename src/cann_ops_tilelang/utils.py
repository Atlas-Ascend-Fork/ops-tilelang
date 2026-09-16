"""Small dependency-free helpers shared by operator wrappers."""


def ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError(f'divisor must be positive, got {divisor}.')
    return (value + divisor - 1) // divisor


def align_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment
