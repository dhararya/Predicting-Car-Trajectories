import numpy as np
from shapely.geometry import Polygon
import shapely.geometry
import math
import torch 
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
    # https://arxiv.org/pdf/1911.08287.pdf 
    M, N = bboxes1.shape[0], bboxes2.shape[0]
    iou_mat = np.zeros((M, N))
    for i in range(0,M):
        m = rotate_box(bboxes1[i])
        h1,w1 = bboxes1[i][2], bboxes1[i][3]
        for j in range(0,N):
            n = rotate_box(bboxes2[j])
            h2,w2 = bboxes2[j][2], bboxes2[j][3]
            iou = m.intersection(n).area / m.union(n).area
            di = distance(m,n)
            v = (4 / (math.pi ** 2)) * math.pow((np.arctan(h2 / w2) - np.arctan(h1 / w1)), 2)
            if(iou >= 0.5):
                alpha = (v/((1-iou)+v))
                iou_mat[i][j]= iou + di + (alpha * v)
            else:
                iou_mat[i][j]= iou + di
    return iou_mat

def distance(box1: Polygon, box2: Polygon):
    box1_center_x, box1_center_y = list(box1.centroid.coords)[0]
    box2_center_x, box2_center_y = list(box2.centroid.coords)[0]
    centriod_distance = (box2_center_x - box1_center_x)**2 + (box2_center_y - box1_center_y)**2
    min1x, min1y, max1x, max1y = box1.bounds
    min2x, min2y, max2x, max2y = box2.bounds
    minx, miny, maxx, maxy = min(min1x,min2x), min(min1y,min2y), max(max1x,max2x), max(max1y,max2y)
    c_val = max(0, (maxx - minx)) ** 2 + max(0, maxy - miny) ** 2
    return centriod_distance / c_val