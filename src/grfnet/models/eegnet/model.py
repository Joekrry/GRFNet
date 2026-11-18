"""EEGNet baseline (Lawhern et al., 2018), via braindecode."""

from __future__ import annotations

from .._braindecode import BraindecodeModel


class EEGNet(BraindecodeModel):
    """EEGNet-8,2 from braindecode with its default hyperparameters.

    Parameters
    ----------
    n_channels, n_times, n_classes : int
        Input channels, input length and number of classes.
    **kwargs
        Passed to ``braindecode.models.EEGNet``, e.g. ``kernel_length`` or ``drop_prob``.

    References
    ----------
    Lawhern et al. (2018). EEGNet: a compact convolutional neural network for EEG-based
    brain-computer interfaces. Journal of Neural Engineering, 15(5), 056013.
    """

    _braindecode_name = "EEGNet"
