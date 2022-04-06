import torch
from torch import Tensor, nn

from dataclasses import dataclass, field
from typing import List, Tuple

from prediction.model import PredictionModel, PredictionModelConfig
from prediction.types import Trajectories
from prediction.utils.transform import transform_using_actor_frame_gauss
from prediction.utils.reshape import flatten, unflatten_batch


class ProbabilisticMLP(nn.Module):
    ## write the code for the new model here
    def __init__(self, config: PredictionModelConfig) -> None:
        super().__init__()
        self._encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(30, 128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.ELU(),
        )

        self._decoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, 50),
                nn.ELU(),
            )


class ProbabilisticModel(PredictionModel):
    def __init__(self, config: PredictionModelConfig) -> None:
        super().__init__(config)
        self._encoder = ProbabilisticMLP(PredictionModelConfig)._encoder
        self._decoder = ProbabilisticMLP(PredictionModelConfig)._decoder

    @staticmethod
    def _postprocess(
        out: Tensor, batch_ids: Tensor, original_x_pose: Tensor
    ) -> List[Tensor]:
        """Postprocess predictions

        1. Unflatten time and position dimensions
        2. Transform predictions back into SDV frame
        3. Unflatten batch and actor dimension

        Args:
            out (Tensor): predicted input trajectories [batch_size * N x T * 2]
            batch_ids (Tensor): id of each actor's batch in the flattened list [batch_size * N]
            original_x_pose (Tensor): original position and yaw of each actor at the latest timestep in SDV frame
                [batch_size * N, 3]

        Returns:
            List[Tensor]: List of length batch_size of output predicted trajectories in SDV frame [N x T x 2]
        """
        num_actors = len(batch_ids)
        out = out.reshape(num_actors, -1, 5)  # [batch_size * N x T x 6]
        # Transform from actor frame, to make the prediction problem easier
        transformed_out = transform_using_actor_frame_gauss(
            out, original_x_pose, translate_to=False
        )

        # Translate so that latest timestep for each actor is the origin
        out_batches = unflatten_batch(transformed_out, batch_ids)
        return out_batches