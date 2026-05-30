Usage
=====

Both generators return ``(X, y)`` pairs of :class:`torch.Tensor` objects that
plug directly into a PyTorch :class:`~torch.utils.data.DataLoader`.

Classification
--------------

:func:`~artificial_dataset.make_classification` produces a labelled multi-class
dataset.  Each class follows a distinct combination of signal components.

.. code-block:: python

   from artificial_dataset import make_classification

   X, y = make_classification(
       n_samples=1000,
       n_classes=2,
       noise_std=0.1,
       random_state=42,
   )
   # X: shape (1000, 2)  — column 0 is x, column 1 is the noisy signal
   # y: shape (1000,)    — integer class labels in [0, n_classes)

Anomaly detection
-----------------

:func:`~artificial_dataset.make_anomaly_dataset` produces a dataset of normal
samples (label ``0``) and anomalous samples (label ``1``).  Anomalies deviate
from the clean signal by a large noise term controlled by ``anomaly_scale``.

.. code-block:: python

   from artificial_dataset import make_anomaly_dataset

   X, y = make_anomaly_dataset(
       n_samples=1000,
       anomaly_fraction=0.05,
       noise_std=0.05,
       anomaly_scale=6.0,
       random_state=42,
   )
   # X: shape (1000, 2)  — column 0 is x, column 1 is the observed signal
   # y: shape (1000,)    — 0 for normal, 1 for anomaly

Custom signal shapes
--------------------

Both generators accept a ``class_params`` / ``signal_params`` argument that
controls which signal components are superimposed.  The ``params`` dict may
contain any combination of ``"linear"``, ``"polynomial"``, and ``"sinusoidal"``
keys:

.. code-block:: python

   import math
   from artificial_dataset import make_classification

   class_params = [
       {
           "sinusoidal": {"amplitude": 1.0, "frequency": 1.0, "phase": 0.0},
           "linear":     {"slope": 0.2, "intercept": 0.0},
       },
       {
           "polynomial": {"coefficients": [0.0, 0.0, 0.1]},
           "sinusoidal": {"amplitude": 0.5, "frequency": 3.0, "phase": math.pi},
       },
   ]

   X, y = make_classification(n_samples=500, class_params=class_params, random_state=0)

Low-level components
--------------------

The primitive building blocks are also exported directly and can be composed
manually using :func:`~artificial_dataset.compose`:

.. code-block:: python

   import torch
   from artificial_dataset import compose, gaussian_noise

   x = torch.linspace(0, 2 * torch.pi, steps=200)

   signal = compose(x, {
       "sinusoidal": {"amplitude": 1.0, "frequency": 2.0, "phase": 0.0},
       "linear":     {"slope": 0.1, "intercept": -0.5},
   })
   noisy = signal + gaussian_noise(x.shape, mean=0.0, std=0.05)
