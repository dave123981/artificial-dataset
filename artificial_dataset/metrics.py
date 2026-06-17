"""Classifier evaluation metrics."""

from typing import Self

import torch


class ClassifierMetrics:
    """Evaluation metrics for a classifier.

    Computes accuracy, macro-averaged precision, recall, F1 score, and a
    confusion matrix from predicted and ground-truth class labels.

    Two construction modes are supported:

    * Constructor — supply full ``y_true`` / ``y_pred`` label tensors directly.
    * :meth:`from_anomaly_indices` — supply the index positions of the positive
      (anomalous) samples as a :class:`torch.Tensor` or :class:`list`; binary
      label vectors are built internally.

    All per-class metrics are macro-averaged over every class that appears in
    either *y_true* or *y_pred*.

    Parameters
    ----------
    y_true : torch.Tensor, shape (n_samples,)
        Ground-truth class labels, dtype ``torch.long``.
    y_pred : torch.Tensor, shape (n_samples,)
        Predicted class labels, dtype ``torch.long``.

    Raises
    ------
    ValueError
        If *y_true* and *y_pred* differ in length, are not 1-D, or are empty.

    Examples
    --------
    >>> import torch
    >>> y_true = torch.tensor([0, 1, 0, 1, 0])
    >>> y_pred = torch.tensor([0, 1, 1, 1, 0])
    >>> m = ClassifierMetrics(y_true, y_pred)
    >>> round(m.accuracy, 2)
    0.8
    >>> m.confusion_matrix
    tensor([[2, 1],
            [0, 2]])
    """

    def __init__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        if y_true.ndim != 1 or y_pred.ndim != 1:
            raise ValueError("y_true and y_pred must be 1-D tensors")
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"y_true and y_pred must have the same length, "
                f"got {y_true.shape[0]} and {y_pred.shape[0]}"
            )
        if y_true.numel() == 0:
            raise ValueError("y_true and y_pred must not be empty")

        self._y_true = y_true.long()
        self._y_pred = y_pred.long()
        self._classes: torch.Tensor = torch.cat([self._y_true, self._y_pred]).unique()
        self._n_classes = int(self._classes.numel())
        self._n_samples = int(self._y_true.numel())

        n = self._n_classes
        true_idx = torch.searchsorted(self._classes, self._y_true)
        pred_idx = torch.searchsorted(self._classes, self._y_pred)
        self._cm: torch.Tensor = torch.bincount(
            true_idx * n + pred_idx, minlength=n * n
        ).reshape(n, n)

    @classmethod
    def from_anomaly_indices(
        cls,
        n_samples: int,
        true_indices: torch.Tensor | list[int],
        pred_indices: torch.Tensor | list[int],
    ) -> Self:
        """Create metrics from anomaly index positions.

        Both *true_indices* and *pred_indices* specify the positions of the
        positive (anomaly) class.  Binary label vectors of length *n_samples*
        are constructed: ``0`` for normal and ``1`` for anomalous.

        Parameters
        ----------
        n_samples : int
            Total number of samples.
        true_indices : torch.Tensor or list[int]
            Index positions of the true anomalies.
        pred_indices : torch.Tensor or list[int]
            Index positions of the predicted anomalies.

        Returns
        -------
        ClassifierMetrics
            Backed by binary label vectors derived from the index positions.

        Raises
        ------
        ValueError
            If *n_samples* is not positive or any index is outside
            ``[0, n_samples)``.

        Examples
        --------
        >>> import torch
        >>> m = ClassifierMetrics.from_anomaly_indices(
        ...     n_samples=5,
        ...     true_indices=[1, 3],
        ...     pred_indices=torch.tensor([1, 2]),
        ... )
        >>> round(m.accuracy, 2)
        0.6
        """
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        true_t = (
            torch.as_tensor(true_indices, dtype=torch.long)
            if isinstance(true_indices, list)
            else true_indices.long()
        )
        pred_t = (
            torch.as_tensor(pred_indices, dtype=torch.long)
            if isinstance(pred_indices, list)
            else pred_indices.long()
        )

        for name, idx in [("true_indices", true_t), ("pred_indices", pred_t)]:
            if idx.numel() > 0 and (int(idx.min()) < 0 or int(idx.max()) >= n_samples):
                raise ValueError(f"{name} contains values outside [0, {n_samples})")

        y_true = torch.zeros(n_samples, dtype=torch.long)
        y_pred = torch.zeros(n_samples, dtype=torch.long)
        if true_t.numel() > 0:
            y_true[true_t] = 1
        if pred_t.numel() > 0:
            y_pred[pred_t] = 1

        return cls(y_true, y_pred)

    @property
    def accuracy(self) -> float:
        """Fraction of correctly classified samples.

        Returns
        -------
        float
            Accuracy in ``[0, 1]``.
        """
        return float((self._y_true == self._y_pred).sum()) / self._n_samples

    @property
    def precision(self) -> float:
        """Macro-averaged precision across all classes.

        Precision for class *c* is ``TP_c / (TP_c + FP_c)``.  Classes with
        no predicted samples contribute ``0``.

        Returns
        -------
        float
            Macro-averaged precision in ``[0, 1]``.
        """
        values: list[float] = []
        for i in range(self._n_classes):
            tp = float(self._cm[i, i])
            denom = float(self._cm[:, i].sum())
            values.append(tp / denom if denom > 0.0 else 0.0)
        return sum(values) / len(values)

    @property
    def recall(self) -> float:
        """Macro-averaged recall across all classes.

        Recall for class *c* is ``TP_c / (TP_c + FN_c)``.  Classes with no
        true samples contribute ``0``.

        Returns
        -------
        float
            Macro-averaged recall in ``[0, 1]``.
        """
        values: list[float] = []
        for i in range(self._n_classes):
            tp = float(self._cm[i, i])
            denom = float(self._cm[i, :].sum())
            values.append(tp / denom if denom > 0.0 else 0.0)
        return sum(values) / len(values)

    @property
    def f1_score(self) -> float:
        """Macro-averaged F1 score across all classes.

        F1 for class *c* is the harmonic mean of its precision and recall.
        Classes where both are ``0`` contribute ``0``.

        Returns
        -------
        float
            Macro-averaged F1 score in ``[0, 1]``.
        """
        values: list[float] = []
        for i in range(self._n_classes):
            tp = float(self._cm[i, i])
            col_sum = float(self._cm[:, i].sum())
            row_sum = float(self._cm[i, :].sum())
            p = tp / col_sum if col_sum > 0.0 else 0.0
            r = tp / row_sum if row_sum > 0.0 else 0.0
            denom = p + r
            values.append(2.0 * p * r / denom if denom > 0.0 else 0.0)
        return sum(values) / len(values)

    @property
    def confusion_matrix(self) -> torch.Tensor:
        """Confusion matrix of shape ``(n_classes, n_classes)``.

        Entry ``[i, j]`` counts samples whose true class is ``classes[i]``
        and predicted class is ``classes[j]``, where *classes* are the sorted
        unique labels seen across *y_true* and *y_pred*.

        Returns
        -------
        torch.Tensor, shape (n_classes, n_classes)
            Confusion matrix, dtype ``torch.long``.
        """
        return self._cm.clone()
