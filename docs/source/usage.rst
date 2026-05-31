Usage
=====

Both generators return :class:`torch.Tensor` objects that plug directly into a
PyTorch :class:`~torch.utils.data.DataLoader`.  The anomaly generator returns
a three-tuple ``(x, y, labels)``; the classification generator returns a
two-tuple ``(X, y)``.

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

The function returns three tensors:

* **x** — shape ``(n_samples,)``: sampled input coordinates, intended for
  plotting only and not required during classification.
* **y** — shape ``(n_samples, n_channels)``: observed signal values, one
  column per channel.
* **labels** — shape ``(n_samples,)``, dtype ``torch.long``: ``0`` for normal
  samples, ``1`` for anomalies.

.. code-block:: python

   from artificial_dataset import make_anomaly_dataset

   x, y, labels = make_anomaly_dataset(
       n_samples=1000,
       anomaly_fraction=0.05,
       noise_std=0.05,
       anomaly_scale=6.0,
       random_state=42,
   )
   # x:      shape (1000,)     — input coordinates for plotting
   # y:      shape (1000, 1)   — observed signal (one channel by default)
   # labels: shape (1000,)     — 0 for normal, 1 for anomaly

Pass a list with more than one entry to ``channel_params`` to get multiple
channels.  Anomalies are injected with the same scaled noise on every channel,
so the label is consistent across the whole row:

.. code-block:: python

   channel_params = [
       {"sinusoidal": {"amplitude": 1.0, "frequency": 1.0, "phase": 0.0}},
       {"linear":     {"slope": 0.3, "intercept": 0.5}},
   ]

   x, y, labels = make_anomaly_dataset(
       n_samples=600,
       anomaly_fraction=0.08,
       channel_params=channel_params,
       random_state=0,
   )
   # y: shape (600, 2)  — two independent signal channels

Custom signal shapes
--------------------

Both generators accept a ``class_params`` / ``channel_params`` argument that
controls which signal components are superimposed.  Each params dict may
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

Evaluating a classifier
-----------------------

:class:`~artificial_dataset.ClassifierMetrics` computes accuracy,
macro-averaged precision, recall, F1 score, and a confusion matrix.  Two
construction modes are available.

**From full label vectors** — pass ``y_true`` and ``y_pred`` directly:

.. code-block:: python

   import torch
   from artificial_dataset import ClassifierMetrics

   y_true = torch.tensor([0, 1, 0, 1, 0, 1])
   y_pred = torch.tensor([0, 1, 1, 1, 0, 0])

   m = ClassifierMetrics(y_true, y_pred)
   print(m.accuracy)          # fraction of correct predictions
   print(m.precision)         # macro-averaged precision
   print(m.recall)            # macro-averaged recall
   print(m.f1_score)          # macro-averaged F1
   print(m.confusion_matrix)  # torch.Tensor shape (n_classes, n_classes)

**From anomaly index positions** — pass the index positions of the positive
(anomaly) class instead of building the full label vectors manually:

.. code-block:: python

   import torch
   from artificial_dataset import ClassifierMetrics, make_anomaly_dataset

   x, y, labels = make_anomaly_dataset(n_samples=500, anomaly_fraction=0.1, random_state=0)

   true_indices = (labels == 1).nonzero(as_tuple=True)[0]
   pred_indices = torch.tensor([3, 27, 88, 142])  # your model's detections

   m = ClassifierMetrics.from_anomaly_indices(
       n_samples=500,
       true_indices=true_indices,   # torch.Tensor or list[int]
       pred_indices=pred_indices,
   )
   print(m.f1_score)
   print(m.confusion_matrix)

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
