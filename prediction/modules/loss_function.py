from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor, nn


def compute_l1_loss(targets: Tensor, predictions: Tensor) -> Tensor:
    """Compute the mean absolute error (MAE)/L1 loss between `predictions` and `targets`.

    Specifically, the l1-weighted MSE loss can be computed as follows:
    1. Compute a binary mask of the `targets` that are not NaN, and apply it to the `targets` and `predictions`
    2. Compute the MAE loss between `predictions` and `targets`.
        This should give us a [batch_size * num_actors x T x 2] tensor `l1_loss`.
    3. Compute the mean of `l1_loss`. This gives us our final scalar loss.

    Args:
        targets: A [batch_size * num_actors x T x 2] tensor, containing the ground truth targets.
        predictions: A [batch_size * num_actors x T x 2] tensor, containing the predictions.

    Returns:
        A scalar MAE loss between `predictions` and `targets`
    """
    mask = 1-targets.isnan().long()
    targets = targets.nan_to_num(0)
    predictions = predictions * mask
    loss = nn.L1Loss(reduction='sum')

    return loss(predictions, targets)

def compute_NLL_loss(targets: Tensor, predictions: Tensor) -> Tensor:
    """Compute the NLL loss between `predictions` and `targets`.
    Returns:
        A scalar NLL loss between `predictions` and `targets`
    """
    means = predictions[:, :, 0:2].view(predictions.shape[0], predictions.shape[1], 2, 1)
    targets =  targets.view(predictions.shape[0], predictions.shape[1], 2, 1)
    mask = 1-targets.isnan().long()
    targets = targets.nan_to_num(0)
    means = means * mask
    cov = predictions[:, :, 2:6].view(predictions.shape[0], predictions.shape[1], 2,2)
    log_det_cov = torch.log(torch.linalg.det(cov)) * torch.squeeze(mask[:, :, 0, 0])
    inv_cov =  torch.linalg.inv(cov) * torch.cat((mask, mask), 3)
    loss = 0.5*log_det_cov+0.5*torch.squeeze(torch.matmul(torch.matmul(torch.transpose(targets-means, 2,3), inv_cov), targets-means))
    loss = torch.squeeze(torch.sum(loss, dim=1))
    return torch.mean(loss)

@dataclass
class PredictionLossConfig:
    """Prediction loss function configuration.

    Attributes:
        l1_loss_weight: The multiplicative weight of the L1 loss
    """

    l1_loss_weight: float


@dataclass
class PredictionLossMetadata:
    """Detailed breakdown of the Prediction loss."""

    total_loss: torch.Tensor
    l1_loss: torch.Tensor


class PredictionLossFunction(torch.nn.Module):
    """A loss function to train a Prediction model."""

    def __init__(self, config: PredictionLossConfig) -> None:
        super(PredictionLossFunction, self).__init__()
        self._l1_loss_weight = config.l1_loss_weight

    def forward(
        self, predictions: List[Tensor], targets: List[Tensor]
    ) -> Tuple[torch.Tensor, PredictionLossMetadata]:
        """Compute the loss between the predicted Predictions and target labels.

        Args:
            predictions: A list of batch_size x [num_actors x T x 2] tensor containing the outputs of
                `PredictionModel`.
            targets:  A list of batch_size x [num_actors x T x 2] tensor containing the ground truth output.

        Returns:
            The scalar tensor containing the weighted loss between `predictions` and `targets`.
        """
        predictions_tensor = torch.cat(predictions)
        targets_tensor = torch.cat(targets)

        # 1. Unpack the targets tensor.
        target_centroids = targets_tensor[..., :2]  # [batch_size * num_actors x T x 2]

        # 2. Unpack the predictions tensor.
        predicted_centroids = predictions_tensor[
            ..., :2
        ]  # [batch_size * num_actors x T x 2]

        # 3. Compute individual loss terms for l1
        l1_loss = compute_l1_loss(target_centroids, predicted_centroids)

        # 4. Aggregate losses using the configured weights.
        total_loss = l1_loss * self._l1_loss_weight

        loss_metadata = PredictionLossMetadata(total_loss, l1_loss)
        return total_loss, loss_metadata


class ProbabalisticPredictionLossFunction(PredictionLossFunction):
    """A loss function to train a Prediction model."""

    def __init__(self, config: PredictionLossConfig) -> None:
        super(PredictionLossFunction, self).__init__()
        self._loss_weight = config.l1_loss_weight

    def forward(
        self, predictions: List[Tensor], targets: List[Tensor]
    ) -> Tuple[torch.Tensor, PredictionLossMetadata]:
        """Compute the loss between the predicted Predictions and target labels.

        Args:
            predictions: A list of batch_size x [num_actors x T x 2] tensor containing the outputs of
                `PredictionModel`.
            targets:  A list of batch_size x [num_actors x T x 2] tensor containing the ground truth output.

        Returns:
            The scalar tensor containing the weighted loss between `predictions` and `targets`.
        """
        predictions_tensor = torch.cat(predictions)
        targets_tensor = torch.cat(targets)

        # 1. Unpack the targets tensor.
        target_centroids = targets_tensor[..., :2]  # [batch_size * num_actors x T x 2]

        # 2. Unpack the predictions tensor.
        predicted_centroids = predictions_tensor[
            ..., :6
        ]  # [batch_size * num_actors x T x 6]
        # 3. Compute individual loss terms for l1
        nll_loss = compute_NLL_loss(target_centroids, predicted_centroids)

        # 4. Aggregate losses using the configured weights.
        total_loss = nll_loss * self._loss_weight

        loss_metadata = PredictionLossMetadata(total_loss, nll_loss)
        return total_loss, loss_metadata
