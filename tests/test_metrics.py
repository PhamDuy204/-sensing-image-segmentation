import torch

from oemseg.metrics.segmentation import ConfusionMatrix, batch_confusion_matrices, flatten_metrics


def test_perfect_prediction_has_unit_metrics_and_test_oa():
    target = torch.tensor([[[0, 1], [2, 3]]])
    matrix = ConfusionMatrix(classes=9)
    matrix.update(target, target)
    metrics = matrix.compute()
    flattened = flatten_metrics("test", 0.25, metrics)
    assert metrics.oa == 1.0
    assert metrics.miou == 1.0
    assert flattened["test_oa"] == 1.0
    assert "test_acc" not in flattened


def test_confusion_matrix_stays_on_prediction_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = torch.tensor([[[0, 1], [1, 1]]], device=device)
    prediction = target.clone()
    matrix = ConfusionMatrix(classes=2, device=device)
    matrix.update(prediction, target)
    assert matrix.matrix.device == target.device


def test_batch_confusion_matrices_preserve_each_sample():
    target = torch.tensor([[[0, 1]], [[1, 1]]])
    prediction = torch.tensor([[[0, 1]], [[0, 1]]])
    matrices = batch_confusion_matrices(prediction, target, classes=2)
    assert matrices.shape == (2, 2, 2)
    assert matrices[0].tolist() == [[1, 0], [0, 1]]
    assert matrices[1].tolist() == [[0, 0], [1, 1]]
