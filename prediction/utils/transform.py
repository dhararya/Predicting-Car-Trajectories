import torch
from torch import Tensor


def transform_using_actor_frame(
    positions: Tensor, actor_frame: Tensor, translate_to: bool = True
):
    """Transform a batch of positions either to or from actor frame

    Args:
        positions (Tensor): [N x T x 2] (x, y)
        actor_pose (Tensor): [N x 3] (x, y, yaw)

    Returns:
        Tensor: [N x T x 2] (x, y) in actor frame
    """
    num_timesteps = positions.shape[1]

    yaw = actor_frame[:, 2]
    if translate_to:
        # Negative rotation to counter the existing rotation on the actor
        yaw = -yaw
    sin_yaw = yaw.sin()
    cos_yaw = yaw.cos()
    rot_mat = torch.stack(
        [
            torch.stack([cos_yaw, -sin_yaw], dim=1),
            torch.stack([sin_yaw, cos_yaw], dim=1),
        ],
        dim=1,
    )  #  N x 2 x 2
    expanded_rot_mat = rot_mat.repeat_interleave(
        repeats=num_timesteps, dim=0
    )  # N * T x 2 x 2

    if translate_to:
        # Move to origin
        translated_positions = (positions - actor_frame[:, None, :2]).reshape(
            -1, 2, 1
        )  # N * T x 2 x 1
        transformed_positions = torch.bmm(
            expanded_rot_mat, translated_positions
        )  # N * T x 2 x 1
        transformed_positions = transformed_positions.reshape(-1, num_timesteps, 2)
    else:
        rotated_positions = torch.bmm(
            expanded_rot_mat, positions.reshape(-1, 2, 1)
        )  # N * T x 2 x 1
        rotated_positions = rotated_positions.reshape(-1, num_timesteps, 2)  # N x T x 2
        transformed_positions = rotated_positions + actor_frame[:, None, :2]

    return transformed_positions.reshape(-1, num_timesteps, 2)


def transform_using_actor_frame_gauss(
    output: Tensor, actor_frame: Tensor, translate_to: bool = True
):
    """Transform a batch of positions either to or from actor frame

    Args:
        output (Tensor): [N x T x 6] (x, y)
        actor_pose (Tensor): [N x 3] (x, y, yaw)

    Returns:
        Tensor: [N x T x 2] (x, y) in actor frame
    """
    positions = output[:, :, 0:1]
    cov = output[:, :, 2:6].reshape(position.shape[0], position.shape[1], 2,2)
    num_timesteps = positions.shape[1]

    yaw = actor_frame[:, 2]
    if translate_to:
        # Negative rotation to counter the existing rotation on the actor
        yaw = -yaw
    sin_yaw = yaw.sin()
    cos_yaw = yaw.cos()
    rot_mat = torch.stack(
        [
            torch.stack([cos_yaw, -sin_yaw], dim=1),
            torch.stack([sin_yaw, cos_yaw], dim=1),
        ],
        dim=1,
    )  #  N x 2 x 2
    expanded_rot_mat = rot_mat.repeat_interleave(
        repeats=num_timesteps, dim=0
    )  # N * T x 2 x 2

    if translate_to:
        # Move to origin
        translated_positions = (positions - actor_frame[:, None, :2]).reshape(
            -1, 2, 1
        )  # N * T x 2 x 1
        transformed_positions = torch.bmm(
            expanded_rot_mat, translated_positions
        )  # N * T x 2 x 1
        transformed_positions = transformed_positions.reshape(-1, num_timesteps, 2)
    else:
        rotated_positions = torch.bmm(
            expanded_rot_mat, positions.reshape(-1, 2, 1)
        )  # N * T x 2 x 1
        rotated_positions = rotated_positions.reshape(-1, num_timesteps, 2)  # N x T x 2
        transformed_positions = rotated_positions + actor_frame[:, None, :2]
    
    rot_cov = torch.bmm(torch.bmm(expanded_rot_mat, cov), torch.transpose(expanded_rot_mat, 2,3)).reshape(position.shape[0], position.shape[1], 4)

    result = np.zeros(position.shape[0], position.shape[1], 6)
    result[:, :, 0:2] = transformed_positions.reshape(-1, num_timesteps, 2)
    result[:, :, 2:6] = rot_cov

    return result

