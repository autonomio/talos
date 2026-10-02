"""Public helpers; framework imports are deferred until use."""
from ..metrics import keras_metrics as metrics
from ..model.early_stopper import early_stopper
from ..model.hidden_layers import hidden_layers
from ..model.normalizers import lr_normalizer
from . import gpu_utils
from .generator import generator
from .power_draw_append import power_draw_append
from .rescale_meanzero import rescale_meanzero
from .sequence_generator import SequenceGenerator
from .torch_history import TorchHistory


def val_split(x, y, split, shuffled=True, *, multi_input=False, seed=None):
    import numpy as np

    from .validation_split import _size, _take
    if not 0 < split < 1:
        raise ValueError('split must lie strictly between zero and one.')
    nested_y = isinstance(y, list)
    size = _size(y, nested_y)
    indices = np.random.default_rng(seed).permutation(size) if shuffled else np.arange(size)
    limit = int(size * (1 - split))
    return (_take(x, indices[:limit], multi_input), _take(y, indices[:limit], nested_y), _take(x, indices[limit:], multi_input), _take(y, indices[limit:], nested_y))


def recover_best_model(*args, **kwargs):
    from .recover_best_model import recover_best_model as recover
    return recover(*args, **kwargs)


__all__ = ['SequenceGenerator', 'TorchHistory', 'early_stopper',
    'generator', 'gpu_utils', 'hidden_layers', 'lr_normalizer', 'metrics',
    'power_draw_append', 'recover_best_model',
    'rescale_meanzero', 'val_split', ]
