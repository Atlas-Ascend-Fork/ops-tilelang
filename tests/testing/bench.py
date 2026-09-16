import os
import sys


class EmptySuppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class SuppressStdoutStderr:
    def __init__(self):
        self._stdout_sink = None
        self._stderr_sink = None
        self._stdout_fd = None
        self._stderr_fd = None
        self._saved_stdout_fd = None
        self._saved_stderr_fd = None
        self._saved_stdout = None
        self._saved_stderr = None

    def __enter__(self):
        self._stdout_sink = open(os.devnull, 'w')
        self._stderr_sink = open(os.devnull, 'w')

        self._saved_stdout = sys.stdout
        self._saved_stderr = sys.stderr
        self._stdout_fd = self._saved_stdout.fileno()
        self._stderr_fd = self._saved_stderr.fileno()

        self._saved_stdout_fd = os.dup(self._stdout_fd)
        self._saved_stderr_fd = os.dup(self._stderr_fd)

        os.dup2(self._stdout_sink.fileno(), self._stdout_fd)
        os.dup2(self._stderr_sink.fileno(), self._stderr_fd)

        sys.stdout = self._stdout_sink
        sys.stderr = self._stderr_sink
        return self

    def __exit__(self, *_exit_args):
        sys.stdout = self._saved_stdout
        sys.stderr = self._saved_stderr

        os.dup2(self._saved_stdout_fd, self._stdout_fd)
        os.dup2(self._saved_stderr_fd, self._stderr_fd)

        os.close(self._saved_stderr_fd)
        os.close(self._saved_stdout_fd)

        self._stderr_sink.close()
        self._stdout_sink.close()


def get_cast_params(params: dict) -> dict:
    cast_param_keys = ['round_sf', 'use_packed_ue8m0', 'use_tma_aligned_col_major_sf']
    return {key: params[key] for key in cast_param_keys if key in params}


_SHORT_NAME = {
    'num_ep_ranks': 'ep',
    'num_experts': 'experts',
    'use_tma_aligned_col_major_sf': 'col',
    'use_packed_ue8m0': 'ue8m0',
    'round_sf': 'round',
}
_LONG_NAME = {short_name: long_name for long_name, short_name in _SHORT_NAME.items()}

_DISPLAY_WIDTH = {
    'num_tokens': 5,
    'num_ep_ranks': 2,
    'num_experts': 3,
    'hidden': 4,
    'use_tma_aligned_col_major_sf': 1,
    'use_packed_ue8m0': 1,
    'round_sf': 1,
    'num_per_channels': 4,
    'numel': 9,
    'num_rows': 7,
    'head_dim': 3,
    'nesterov': 5,
}


def make_param_id(params: dict, display: bool = False) -> str:
    return ''.join(
        _format_param_id_field(_SHORT_NAME.get(key, key), str(value), display)
        for key, value in params.items()
        if value is not None
    )


def format_param_id(param_id: str, display: bool = False) -> str:
    fields = [field for field in param_id.split(',') if field]
    return ''.join(
        _format_param_id_field(name, value, display) for field in fields for name, _, value in (field.partition('='),)
    )


def _format_param_id_field(name: str, value: str, display: bool) -> str:
    field = f'{name}={value},'
    if not display:
        return field

    width = _display_param_value_width(name, value)
    if width is None:
        return field
    return f'{field:<{len(name) + 1 + width + 1}}'


def _display_param_value_width(name: str, value: str) -> int | None:
    width = _DISPLAY_WIDTH.get(_LONG_NAME.get(name, name))
    if width is None:
        return None
    if value in ('True', 'False'):
        width = 5
    return max(width, len(value))
