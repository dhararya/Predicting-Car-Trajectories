import numpy as np
from shapely.geometry import Polygon
import math

def _get_rotated_coordinates(boxes: np.ndarray) -> np.ndarray:
    cos_yaw = np.cos(boxes[:, 4])
    sin_yaw = np.sin(boxes[:, 4])
    ox = boxes[:, 0]
    oy = boxes[:, 1]

    # first coordinate
    x = boxes[:, 0] + boxes[:, 2] / 2
    y = boxes[:, 1] + boxes[:, 3] / 2
    coord_1 = np.stack([ox+cos_yaw * (x-ox) - sin_yaw * (y-oy), oy+ sin_yaw * (x-ox) + cos_yaw * (y-oy)], axis=1)

    # second coordinate
    x = boxes[:, 0] + boxes[:, 2] / 2
    y = boxes[:, 1] - boxes[:, 3] / 2
    coord_2 = np.stack([ox+cos_yaw * (x-ox) - sin_yaw * (y-oy), oy+ sin_yaw * (x-ox) + cos_yaw * (y-oy)], axis=1)

    # third coordinate
    x = boxes[:, 0] - boxes[:, 2] / 2
    y = boxes[:, 1] - boxes[:, 3] / 2
    coord_3 = np.stack([ox+cos_yaw * (x-ox) - sin_yaw * (y-oy), oy+ sin_yaw * (x-ox) + cos_yaw * (y-oy)], axis=1)

    # fourth coordinate
    x = boxes[:, 0] - boxes[:, 2] / 2
    y = boxes[:, 1] + boxes[:, 3] / 2
    coord_4 = np.stack([ox+cos_yaw * (x-ox) - sin_yaw * (y-oy), oy+ sin_yaw * (x-ox) + cos_yaw * (y-oy)], axis=1)

    return np.stack([coord_1, coord_2, coord_3, coord_4], axis=1)

# change opt_cost to 1 to use the sophisticated loss function 
def iou_2d(bboxes1: np.ndarray, bboxes2: np.ndarray, opt_cost=0) -> np.ndarray:
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
        h1,w1 = bboxes1[m][2], bboxes1[m][3]
        for n in range(N):
            if len(n_polygons) < n+1:
                n_polygons.append(Polygon(n_boxes[n]))
            n_polygon = n_polygons[n]
            h2, w2 = bboxes2[n][2], bboxes2[n][3]
            iou = m_polygon.intersection(n_polygon).area / m_polygon.union(n_polygon).area
            dis = distance(m_polygon,n_polygon)
            v = (4 / (math.pi ** 2)) * math.pow((np.arctan(h2 / w2) - np.arctan(h1 / w1)), 2)
            if opt_cost == 0:
                iou_mat[m][n] = iou
            else:
                if(iou >= 0.5):
                    alpha = (v/((1-iou)+v))
                    iou_mat[m][n]= iou + dis + (alpha * v)
                else:
                    iou_mat[m][n] = iou + dis
    return iou_mat

def distance(box1: Polygon, box2: Polygon):
    diff_area = box1.difference(box2)
    box1_center_x, box1_center_y = list(box1.centroid.coords)[0]
    box2_center_x, box2_center_y = list(box2.centroid.coords)[0]
    centriod_distance = ((box2_center_x - box1_center_x)**2) + ((box2_center_y - box1_center_y)**2)
    min1x, min1y, max1x, max1y = box1.bounds
    min2x, min2y, max2x, max2y = box2.bounds
    minx, miny, maxx, maxy = min(min1x,min2x), min(min1y,min2y), max(max1x,max2x), max(max1y,max2y)
    c_val = max(0, (maxx - minx)) ** 2 + max(0, maxy - miny) ** 2
    return centriod_distance / c_val
