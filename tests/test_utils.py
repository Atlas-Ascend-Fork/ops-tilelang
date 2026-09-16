import pytest

from cann_ops_tilelang.utils import align_up, ceil_div


@pytest.mark.parametrize(
    ('value', 'divisor', 'expected'),
    [(0, 8, 0), (1, 8, 1), (8, 8, 1), (9, 8, 2)],
)
def test_ceil_div(value, divisor, expected):
    assert ceil_div(value, divisor) == expected


def test_ceil_div_rejects_non_positive_divisor():
    with pytest.raises(ValueError, match='positive'):
        ceil_div(1, 0)


@pytest.mark.parametrize(('value', 'alignment', 'expected'), [(0, 16, 0), (1, 16, 16), (17, 16, 32)])
def test_align_up(value, alignment, expected):
    assert align_up(value, alignment) == expected
