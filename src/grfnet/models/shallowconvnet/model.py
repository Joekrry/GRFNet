"""ShallowConvNet baseline (Schirrmeister et al., 2017), via braindecode."""

from __future__ import annotations

from .._braindecode import BraindecodeModel


class ShallowConvNet(BraindecodeModel):
    """Shallow ConvNet (braindecode's ``ShallowFBCSPNet``) with its default hyperparameters.

    Parameters
    ----------
    n_channels, n_times, n_classes : int
        Input channels, input length and number of classes.
    **kwargs
        Passed to ``braindecode.models.ShallowFBCSPNet``, e.g. ``filter_time_length`` or ``drop_prob``.

    References
    ----------
    Schirrmeister et al. (2017). Deep learning with convolutional neural networks for EEG decoding
    and visualization. Human Brain Mapping, 38(11), 5391-5420.
    """

    _braindecode_name = "ShallowFBCSPNet"
