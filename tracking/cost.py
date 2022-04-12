import numpy as np
from shapely.geometry import Polygon
import shapely.geometry
import shapely.affinity


def rotate_box(box: np.ndarray):
    length = box[2]
    width = box[3]
    geometryBox = shapely.geometry.box(-length/2, -width/2, length/2, width/2)
    rc = shapely.affinity.rotate(geometryBox, box[4], use_radians=True)
    return shapely.affinity.translate(rc, box[0] , box[1])


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
    for i in range(0,M):
        m = rotate_box(bboxes1[i])
        for j in range(0,N):
            n = rotate_box(bboxes2[j])
            iou_mat[i][j]= m.intersection(n).area / m.union(n).area
    return iou_mat
