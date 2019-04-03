# In this init we load everything under utils in the Talos namespace

try:
    from kerasplotlib import TrainingLog as live
except ImportError:
    print('Matplotlib backend loading failed')

from ..model.normalizers import lr_normalizer
from ..model.layers import hidden_layers
from ..model.early_stopper import early_stopper
from .generator import generator
from . import gpu_utils
import talos.metrics.keras_metrics as metrics
