import numpy as np
from shapely.geometry import Polygon

def _get_rotated_coordinates(boxes: np.ndarray) -> np.ndarray:
    cos_yaw = np.cos(boxes[:, 4])
    sin_yaw = np.sin(boxes[:, 4])

    # first coordinate
    coord_1x = boxes[:, 0] + boxes[:, 3] / 2
    coord_1y = boxes[:, 1] + boxes[:, 2] / 2
    print(coord_1x)
    print(coord_1y)
    coord_1 = np.stack([cos_yaw * coord_1x - sin_yaw * coord_1y, sin_yaw * coord_1x + cos_yaw * coord_1y], axis=1)
    print(coord_1)

    # second coordinate
    coord_2x = boxes[:, 0] + boxes[:, 3] / 2
    coord_2y = boxes[:, 1] - boxes[:, 2] / 2
    coord_2 = np.stack([cos_yaw * coord_2x - sin_yaw * coord_2y, sin_yaw * coord_2x + cos_yaw * coord_2y], axis=1)

    # third coordinate
    coord_3x = boxes[:, 0] - boxes[:, 3] / 2
    coord_3y = boxes[:, 1] - boxes[:, 2] / 2
    coord_3 = np.stack([cos_yaw * coord_3x - sin_yaw * coord_3y, sin_yaw * coord_3x + cos_yaw * coord_3y], axis=1)

    # fourth coordinate
    coord_4x = boxes[:, 0] - boxes[:, 3] / 2
    coord_4y = boxes[:, 1] + boxes[:, 2] / 2
    coord_4 = np.stack([cos_yaw * coord_4x - sin_yaw * coord_4y, sin_yaw * coord_4x + cos_yaw * coord_4y], axis=1)

    return np.stack([coord_1, coord_2, coord_3, coord_4], axis=1)

def iou_2d(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Computes 2D intersection over union of two sets of bounding boxes

    Args:
        bboxes1: bounding box set of shape [M, 5], each row corresponding to x, y, l, w, yaw of the bounding box
        bboxes2: bounding box set of shape [N, 5], each row corresponding to x, y, l, w, yaw of the bounding box
    Returns:
        iou_mat: matrix of shape [M, N], where iou_mat[i, j] is the 2D IoU value between bboxes[i] and bboxes[j].
        You should use the Polygon class from the shapely package to compute the area of intersection/union.
    """
    M, N = bboxes1.shape[0], bboxes2.shape[0]
    iou_mat = np.zeros((M, N))
    m_boxes = _get_rotated_coordinates(bboxes1)
    n_boxes = _get_rotated_coordinates(bboxes2)
    n_polygons = []
    for m in range(M):
        m_polygon = Polygon(m_boxes[m])
        for n in range(N):
            if len(n_polygons) < n+1:
                n_polygons.append(Polygon(n_boxes[n]))
            n_polygon = n_polygons[n]
            if m==1 and n==1:
                print(m_boxes[m])
                print(n_boxes[n])
                print(m_polygon.intersection(n_polygon).area)
            iou_mat[m][n] = m_polygon.intersection(n_polygon).area / m_polygon.union(n_polygon).area
    return iou_mat
