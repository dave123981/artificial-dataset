"""Tests for ClassifierMetrics."""

import pytest
import torch

from artificial_dataset.metrics import ClassifierMetrics

# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------

# y_true = [0, 1, 0, 1, 0], y_pred = [0, 1, 1, 1, 0]
# cm = [[2, 1], [0, 2]]  (TN=2, FP=1, FN=0, TP=2)
_Y_TRUE = torch.tensor([0, 1, 0, 1, 0])
_Y_PRED = torch.tensor([0, 1, 1, 1, 0])


# ------------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------------


def test_accuracy_perfect() -> None:
    """Perfect classifier scores 1.0 accuracy."""
    y = torch.tensor([0, 1, 2, 0, 1])
    m = ClassifierMetrics(y, y.clone())
    assert m.accuracy == pytest.approx(1.0)


def test_accuracy_all_wrong() -> None:
    """All-wrong binary classifier scores 0.0 accuracy."""
    y_true = torch.tensor([0, 1, 0, 1])
    y_pred = torch.tensor([1, 0, 1, 0])
    assert ClassifierMetrics(y_true, y_pred).accuracy == pytest.approx(0.0)


def test_accuracy_partial() -> None:
    """Partial binary classifier accuracy is computed correctly."""
    assert ClassifierMetrics(_Y_TRUE, _Y_PRED).accuracy == pytest.approx(0.8)


# ------------------------------------------------------------------
# Precision
# ------------------------------------------------------------------


def test_precision_perfect() -> None:
    """Perfect classifier scores 1.0 macro precision."""
    y = torch.tensor([0, 1, 0, 1])
    assert ClassifierMetrics(y, y.clone()).precision == pytest.approx(1.0)


def test_precision_partial() -> None:
    """Macro precision for the partial binary example is 5/6."""
    # class 0: TP=2, col_sum=2  → p=1.0
    # class 1: TP=2, col_sum=3  → p=2/3
    # macro = (1.0 + 2/3) / 2 = 5/6
    assert ClassifierMetrics(_Y_TRUE, _Y_PRED).precision == pytest.approx(5 / 6)


def test_precision_all_wrong() -> None:
    """All-wrong binary classifier scores 0.0 macro precision."""
    y_true = torch.tensor([0, 1, 0, 1])
    y_pred = torch.tensor([1, 0, 1, 0])
    assert ClassifierMetrics(y_true, y_pred).precision == pytest.approx(0.0)


# ------------------------------------------------------------------
# Recall
# ------------------------------------------------------------------


def test_recall_perfect() -> None:
    """Perfect classifier scores 1.0 macro recall."""
    y = torch.tensor([0, 1, 0, 1])
    assert ClassifierMetrics(y, y.clone()).recall == pytest.approx(1.0)


def test_recall_partial() -> None:
    """Macro recall for the partial binary example is 5/6."""
    # class 0: TP=2, row_sum=3  → r=2/3
    # class 1: TP=2, row_sum=2  → r=1.0
    # macro = (2/3 + 1.0) / 2 = 5/6
    assert ClassifierMetrics(_Y_TRUE, _Y_PRED).recall == pytest.approx(5 / 6)


def test_recall_all_wrong() -> None:
    """All-wrong binary classifier scores 0.0 macro recall."""
    y_true = torch.tensor([0, 1, 0, 1])
    y_pred = torch.tensor([1, 0, 1, 0])
    assert ClassifierMetrics(y_true, y_pred).recall == pytest.approx(0.0)


# ------------------------------------------------------------------
# F1 score
# ------------------------------------------------------------------


def test_f1_perfect() -> None:
    """Perfect classifier scores 1.0 macro F1."""
    y = torch.tensor([0, 1, 0, 1])
    assert ClassifierMetrics(y, y.clone()).f1_score == pytest.approx(1.0)


def test_f1_partial() -> None:
    """Macro F1 for the partial binary example is 0.8."""
    # class 0: p=1.0, r=2/3 → f1=2*(1.0*2/3)/(1.0+2/3)=4/5=0.8
    # class 1: p=2/3, r=1.0 → f1=4/5=0.8
    # macro = 0.8
    assert ClassifierMetrics(_Y_TRUE, _Y_PRED).f1_score == pytest.approx(0.8)


def test_f1_all_wrong() -> None:
    """All-wrong binary classifier scores 0.0 macro F1."""
    y_true = torch.tensor([0, 1, 0, 1])
    y_pred = torch.tensor([1, 0, 1, 0])
    assert ClassifierMetrics(y_true, y_pred).f1_score == pytest.approx(0.0)


# ------------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------------


def test_confusion_matrix_shape_binary() -> None:
    """Confusion matrix is 2x2 for binary classification."""
    assert ClassifierMetrics(_Y_TRUE, _Y_PRED).confusion_matrix.shape == (2, 2)


def test_confusion_matrix_values_binary() -> None:
    """Confusion matrix entries match manual calculation."""
    cm = ClassifierMetrics(_Y_TRUE, _Y_PRED).confusion_matrix
    expected = torch.tensor([[2, 1], [0, 2]])
    assert torch.equal(cm, expected)


def test_confusion_matrix_shape_multiclass() -> None:
    """Confusion matrix is 3x3 for three-class classification."""
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    cm = ClassifierMetrics(y, y.clone()).confusion_matrix
    assert cm.shape == (3, 3)


def test_confusion_matrix_perfect_is_diagonal() -> None:
    """Perfect classifier produces a diagonal confusion matrix."""
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    cm = ClassifierMetrics(y, y.clone()).confusion_matrix
    assert torch.equal(cm, torch.diag(cm.diag()))


def test_confusion_matrix_dtype() -> None:
    """Confusion matrix has dtype torch.long."""
    cm = ClassifierMetrics(_Y_TRUE, _Y_PRED).confusion_matrix
    assert cm.dtype == torch.long


def test_confusion_matrix_is_clone() -> None:
    """Mutating the returned confusion matrix does not affect internal state."""
    m = ClassifierMetrics(_Y_TRUE, _Y_PRED)
    cm = m.confusion_matrix
    cm[0, 0] = 999
    assert int(m.confusion_matrix[0, 0]) == 2


# ------------------------------------------------------------------
# from_anomaly_indices — list inputs
# ------------------------------------------------------------------


def test_from_anomaly_indices_list_accuracy() -> None:
    """from_anomaly_indices with list inputs computes accuracy correctly."""
    # y_true=[0,1,0,1,0], y_pred=[0,1,1,0,0]: correct at 0,1,4 → 3/5
    m = ClassifierMetrics.from_anomaly_indices(5, [1, 3], [1, 2])
    assert m.accuracy == pytest.approx(0.6)


def test_from_anomaly_indices_list_confusion_matrix() -> None:
    """from_anomaly_indices with list inputs produces the right confusion matrix."""
    # y_true=[0,1,0,1,0], y_pred=[0,1,1,0,0]
    m = ClassifierMetrics.from_anomaly_indices(5, [1, 3], [1, 2])
    expected = torch.tensor([[2, 1], [1, 1]])
    assert torch.equal(m.confusion_matrix, expected)


def test_from_anomaly_indices_list_perfect() -> None:
    """Matching true and pred indices gives accuracy 1.0."""
    m = ClassifierMetrics.from_anomaly_indices(10, [2, 5, 8], [2, 5, 8])
    assert m.accuracy == pytest.approx(1.0)


# ------------------------------------------------------------------
# from_anomaly_indices — tensor inputs
# ------------------------------------------------------------------


def test_from_anomaly_indices_tensor_accuracy() -> None:
    """from_anomaly_indices with tensor inputs computes accuracy correctly."""
    m = ClassifierMetrics.from_anomaly_indices(
        5,
        torch.tensor([1, 3]),
        torch.tensor([1, 2]),
    )
    assert m.accuracy == pytest.approx(0.6)


def test_from_anomaly_indices_mixed_inputs() -> None:
    """from_anomaly_indices accepts one list and one tensor."""
    m = ClassifierMetrics.from_anomaly_indices(5, [1, 3], torch.tensor([1, 2]))
    assert m.accuracy == pytest.approx(0.6)


# ------------------------------------------------------------------
# from_anomaly_indices — empty indices
# ------------------------------------------------------------------


def test_from_anomaly_indices_empty_lists() -> None:
    """Empty index lists mean all samples are normal — accuracy is 1.0."""
    m = ClassifierMetrics.from_anomaly_indices(10, [], [])
    assert m.accuracy == pytest.approx(1.0)


def test_from_anomaly_indices_empty_tensor() -> None:
    """Empty index tensors mean all samples are normal — accuracy is 1.0."""
    m = ClassifierMetrics.from_anomaly_indices(
        10, torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
    )
    assert m.accuracy == pytest.approx(1.0)


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------


def test_shape_mismatch_raises() -> None:
    """ValueError is raised when y_true and y_pred have different lengths."""
    with pytest.raises(ValueError, match="same length"):
        ClassifierMetrics(torch.tensor([0, 1]), torch.tensor([0, 1, 0]))


def test_not_1d_raises() -> None:
    """ValueError is raised when inputs are not 1-D."""
    y = torch.tensor([[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="1-D"):
        ClassifierMetrics(y, y)


def test_empty_raises() -> None:
    """ValueError is raised when the label tensors are empty."""
    with pytest.raises(ValueError, match="empty"):
        ClassifierMetrics(torch.tensor([]), torch.tensor([]))


def test_nonpositive_n_samples_raises() -> None:
    """ValueError is raised when n_samples is not positive."""
    with pytest.raises(ValueError, match="n_samples"):
        ClassifierMetrics.from_anomaly_indices(0, [], [])


def test_true_index_out_of_range_raises() -> None:
    """ValueError is raised when true_indices contains an out-of-range index."""
    with pytest.raises(ValueError, match="true_indices"):
        ClassifierMetrics.from_anomaly_indices(5, [0, 5], [])


def test_pred_index_out_of_range_raises() -> None:
    """ValueError is raised when pred_indices contains an out-of-range index."""
    with pytest.raises(ValueError, match="pred_indices"):
        ClassifierMetrics.from_anomaly_indices(5, [], [-1])
