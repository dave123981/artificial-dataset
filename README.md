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

X, y = make_anomaly_dataset(
    n_samples=1000,
    anomaly_fraction=0.05,
    noise_std=0.05,
    anomaly_scale=6.0,
    random_state=42,
)
# X: torch.Tensor of shape (1000, 2)  — column 0: x values, column 1: observed signal
# y: torch.Tensor of shape (1000,)    — 0 for normal, 1 for anomaly
```

### Custom signal shapes

Both generators accept a `class_params` / `signal_params` argument that
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
