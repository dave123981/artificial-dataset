Usage
=====

Both generators produce :class:`torch.Tensor` data that plugs directly into a
PyTorch :class:`~torch.utils.data.DataLoader`.  The classification generator
returns a two-tuple ``(X, y)``; the anomaly generator returns an
:class:`~artificial_dataset.AnomalyDataset` dataclass whose main field ``y``
has shape ``(m, T)`` — a single multivariate time series of ``m`` channels,
each of length ``T``.

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

:func:`~artificial_dataset.make_anomaly_dataset` produces a single multivariate
time series.  Every channel shares a smooth baseline (a superposition of signal
components) plus Gaussian measurement noise, and a configurable number of
**positive triangular spikes** are added on top.  Anomalies are therefore always
peaks that rise above the baseline, never one-off off-points.

The function returns an :class:`~artificial_dataset.AnomalyDataset` with:

* **y** — shape ``(m, T)``, dtype ``torch.float32``: the main data tensor of
  ``m`` channels, each of length ``T``.
* **labels** — shape ``(T,)``, dtype ``torch.long``: per-timestep anomaly mask,
  ``1`` inside the support of a spike and ``0`` elsewhere.
* **t** — shape ``(T,)``: the time grid the baselines are evaluated on.
* **peak_indices** — a sorted ``LongTensor`` of the ground-truth spike-event
  centres as sample positions into the series.

.. code-block:: python

   from artificial_dataset import make_anomaly_dataset

   data = make_anomaly_dataset(
       series_length=1000,
       noise_std=0.4,
       random_state=42,
   )
   data.y.shape         # torch.Size([1, 1000]) — (m, T)
   data.labels.shape    # torch.Size([1000])    — 0 normal, 1 anomalous timestep
   data.t.shape         # torch.Size([1000])    — time grid
   data.peak_indices    # LongTensor of spike-event centres

Pass a list with more than one entry to ``channel_params`` to get multiple
channels.  Each spike event shares its centre and width across all channels,
while every channel responds with its own random amplitude:

.. code-block:: python

   channel_params = [
       {"sinusoidal": {"amplitude": 1.0, "frequency": 1.0, "phase": 0.0}},
       {"linear":     {"slope": 0.3, "intercept": 0.5}},
   ]

   data = make_anomaly_dataset(
       series_length=600,
       channel_params=channel_params,
       random_state=0,
   )
   # data.y: shape (2, 600)  — two channels

Configuring the spikes
~~~~~~~~~~~~~~~~~~~~~~~~

The spikes are random but fully configurable through the
:class:`~artificial_dataset.SpikeParams` dataclass.  Every property is drawn
uniformly from an inclusive range:

.. code-block:: python

   from artificial_dataset import SpikeParams, make_anomaly_dataset

   spikes = SpikeParams(
       amplitude_range=(4.0, 7.0),  # peak height, sampled per channel
       width_range=(3, 6),          # triangular half-width in samples
       count_range=(3, 8),          # number of spike events in the series
       margin=20,                   # keep spikes this far from either end
   )

   data = make_anomaly_dataset(
       series_length=1000,
       spike_params=spikes,
       random_state=0,
   )

Train / validation / test splitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``split=(train, val, test)`` to receive an
:class:`~artificial_dataset.AnomalySplits` instead of a single dataset.  The
timeline is cut contiguously from the beginning: the first fraction of the ``T``
timesteps becomes ``train``, the next ``val``, and the remainder ``test``.  Each
segment's ``peak_indices`` are filtered to the spikes it contains and re-based to
local sample positions.  The same partition is available as
:meth:`~artificial_dataset.AnomalyDataset.split` on an existing dataset:

.. code-block:: python

   splits = make_anomaly_dataset(
       series_length=1000,
       split=(0.7, 0.15, 0.15),
       random_state=6,
   )
   splits.train.y.shape   # torch.Size([1, 700])
   splits.val.y.shape     # torch.Size([1, 150])
   splits.test.y.shape    # torch.Size([1, 150])

   # Equivalently, split an existing dataset after the fact:
   data = make_anomaly_dataset(series_length=1000, random_state=6)
   splits = data.split((0.7, 0.15, 0.15))

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

   data = make_anomaly_dataset(series_length=1000, random_state=0)

   # Detector flagging timesteps whose value crosses a threshold.
   timestep_max = data.y.amax(dim=0)
   pred_indices = (timestep_max > 2.0).nonzero(as_tuple=True)[0]
   true_indices = (data.labels == 1).nonzero(as_tuple=True)[0]

   m = ClassifierMetrics.from_anomaly_indices(
       n_samples=data.y.shape[1],
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
