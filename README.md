# artificial-dataset

[![coverage](.badges/coverage.svg)](https://github.com/dcintlab/artificial-dataset/actions/workflows/tests.yml)

Artificial dataset generation from Gaussian noise, linear, polynomial, and
sinusoidal signal components.  The package provides two high-level PyTorch
generators for classification and anomaly detection tasks.

## Features

- **Classification** — multi-class labelled dataset where each class follows a
  distinct combination of signal components.
- **Anomaly detection** — normal vs. anomalous samples drawn from the same
  signal, with anomalies injected via amplified Gaussian noise.
- **Classifier evaluation** — `ClassifierMetrics` computes accuracy,
  macro-averaged precision, recall, F1, and a confusion matrix from full label
  vectors or from anomaly index lists / tensors.
- **Composable primitives** — `linear`, `polynomial`, `sinusoidal`, and
  `gaussian_noise` building blocks that can be freely combined via `compose`.
- **PyTorch-native** — all generators return `torch.Tensor` objects that plug
  directly into a `DataLoader`.
- **Reproducible** — every generator accepts a `random_state` seed.

## Installation

Requires Python ≥ 3.12 and PyTorch ≥ 2.12.

```bash
git clone git@github.com:dcintlab/artificial-dataset.git
cd artificial-dataset
pip install -e .
```
Install directly from Github.

```bash
pip install git+https://github.com/intensivedatacomp/artificial-dataset.git
```

## Quick start

### Classification

```python
from artificial_dataset import make_classification

X, y = make_classification(
    n_samples=1000,
    n_classes=2,
    noise_std=0.1,
    random_state=42,
)
# X: torch.Tensor of shape (1000, 2)  — column 0: x values, column 1: noisy signal
# y: torch.Tensor of shape (1000,)    — integer class labels in [0, n_classes)
```

### Anomaly detection

```python
from artificial_dataset import make_anomaly_dataset

data = make_anomaly_dataset(
    series_length=1000,
    noise_std=0.4,
    random_state=42,
)
# data.y:            torch.Tensor of shape (m, T)  — m channels, length T
# data.labels:       torch.Tensor of shape (T,)    — 0 normal, 1 anomalous timestep
# data.t:            torch.Tensor of shape (T,)    — time grid
# data.peak_indices: LongTensor of spike-event centres

# Split along the time axis into train / val / test segments:
splits = make_anomaly_dataset(series_length=1000, split=(0.7, 0.15, 0.15))
# splits.train.y, splits.val.y, splits.test.y  — (m, 700), (m, 150), (m, 150)
```

### Custom signal shapes

Both generators accept a `class_params` / `channel_params` argument that
controls which components are superimposed.  Each entry is a dict with any
combination of `"linear"`, `"polynomial"`, and `"sinusoidal"` keys:

```python
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
```

### Evaluating a classifier

`ClassifierMetrics` accepts full label vectors or, for the anomaly case, the
index positions of the positive class.

```python
import torch
from artificial_dataset import ClassifierMetrics, make_anomaly_dataset

# --- from full label vectors ---
data = make_anomaly_dataset(series_length=500, random_state=0)
y_true = data.labels             # per-timestep mask, shape (T,)
y_pred = y_true.clone()          # replace with your model's predictions
y_pred[::10] = 1 - y_pred[::10] # flip every 10th label to simulate errors

m = ClassifierMetrics(y_true, y_pred)
print(m.accuracy)          # float
print(m.precision)         # macro-averaged
print(m.recall)            # macro-averaged
print(m.f1_score)          # macro-averaged
print(m.confusion_matrix)  # torch.Tensor of shape (n_classes, n_classes)

# --- from anomaly index lists (no need to build the full y vector) ---
true_anomaly_indices = (y_true == 1).nonzero(as_tuple=True)[0]
pred_anomaly_indices = torch.tensor([12, 45, 100])  # model's detections

m2 = ClassifierMetrics.from_anomaly_indices(
    n_samples=data.y.shape[1],
    true_indices=true_anomaly_indices,
    pred_indices=pred_anomaly_indices,
)
```

### Low-level components

The primitives are also exported directly and can be composed manually:

```python
import torch
from artificial_dataset import compose, gaussian_noise

x = torch.linspace(0, 2 * torch.pi, steps=200)

signal = compose(x, {
    "sinusoidal": {"amplitude": 1.0, "frequency": 2.0, "phase": 0.0},
    "linear":     {"slope": 0.1, "intercept": -0.5},
})
noisy = signal + gaussian_noise(x.shape, mean=0.0, std=0.05)
```

## Documentation

Full API reference and usage examples are available in the docs.  Build them
locally with:

```bash
pip install -e ".[docs]"
cd docs && make html
```

The rendered HTML is then available at `docs/build/html/index.html`.

## Contributing

1. Make sure the test suite and pre-commit hooks pass on your branch before
   opening a pull request.
2. Follow the [NumPy docstring convention](https://numpydoc.readthedocs.io/en/latest/format.html).
3. Keep each pull request focused on a single change.

```bash
git switch -c feature/your-awesome-feature

# ... make changes ...

git add .
git commit -m "Useful commit message"
git push --set-upstream origin feature/your-awesome-feature
xdg-open https://github.com/dcintlab/artificial-dataset/pull/new/feature/your-awesome-feature

# ... merge the branch to main, make sure that the pipeline passes, delete the branch ...

git switch main
git pull
git branch -d feature/your-awesome-feature
git branch -d feature/your-awesome-feature --remote
```

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
