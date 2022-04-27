import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from .tracking_utils import kalman_filter

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)

    # if atlbrs == [] or btlbrs == []:
    #     return np.empty((len(atlbrs), len(btlbrs)), dtype=int)
    # _ious = bbox_iou(np.array(atlbrs), np.array(btlbrs), type='IOU', eps=1e-7)

    cost_matrix = 1 - _ious

    return cost_matrix


def bbox_iou(box1, box2, type='IOU', eps=1e-7):
    box1 = box1.T
    box2 = box2.T

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0][:, np.newaxis], box1[1][:, np.newaxis], box1[2][:, np.newaxis], box1[3][:, np.newaxis]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0][np.newaxis, :], box2[1][np.newaxis, :], box2[2][np.newaxis, :], box2[3][np.newaxis, :]

    # Intersection area
    inter = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
            (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union  # IoU
    if type == 'GIOU':
        cw = np.maximum(b1_x2, b2_x2) - np.minimum(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = np.maximum(b1_y2, b2_y2) - np.minimum(b1_y1, b2_y1)  # convex height        
        c_area = cw * ch + eps  # convex area
        iou = iou - (c_area - union) / c_area  # GIoU
    elif type == 'DIOU':
        cw = np.maximum(b1_x2, b2_x2) - np.minimum(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = np.maximum(b1_y2, b2_y2) - np.minimum(b1_y1, b2_y1)  # convex height         
        c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared    
        iou = iou - rho2 / c2  # DIoU
    elif type == 'CIOU':
        cw = np.maximum(b1_x2, b2_x2) - np.minimum(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = np.maximum(b1_y2, b2_y2) - np.minimum(b1_y1, b2_y1)  # convex height         
        c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared    
        v = (4 / math.pi ** 2) * pow(np.arctan(w2 / h2) - np.arctan(w1 / h1), 2)
        alpha = v / (v - iou + (1 + eps))
        iou = iou - (rho2 / c2 + v * alpha)  # CIoU
    elif type == 'EIOU':
        cw = np.maximum(b1_x2, b2_x2) - np.minimum(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = np.maximum(b1_y2, b2_y2) - np.minimum(b1_y1, b2_y1)  # convex height         
        c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared    
        rho3 = (w1-w2) **2
        c3 = cw ** 2 + eps
        rho4 = (h1-h2) **2
        c4 = ch ** 2 + eps
        iou = iou - rho2 / c2 - rho3 / c3 - rho4 / c4  # EIoU

    return iou.clip(0)


def pixel_d(atlbr, btlbr):
    # axis-aligned bounding box
    aAABB = [[atlbr[0], atlbr[1]], [atlbr[0], atlbr[3]], [atlbr[2], atlbr[1]], [atlbr[2], atlbr[3]]] 
    bAABB = [[btlbr[0], btlbr[1]], [btlbr[0], btlbr[3]], [btlbr[2], btlbr[1]], [btlbr[2], btlbr[3]]] 
    aAABB = np.array(aAABB)
    bAABB = np.array(bAABB)
    return np.mean(np.sqrt(np.sum(np.square(bAABB-aAABB), axis=1)))

def pixel_distance(atracks, btracks):
    """
    Compute cost based on mean of 4 tlbr points
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    cost_matrix = np.zeros((len(atracks), len(btracks)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    # small distance means similar pairs.
    cost_matrix =np.abs(cdist(atlbrs, btlbrs, pixel_d))  # Nomalized features by ceiling to 0

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    # small distance means similar pairs.
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features by ceiling to 0
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, logger=None, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    if logger is not None:
        logger.debug('gating threshold: {}'.format(gating_threshold))
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        if logger is not None:
            logger.debug('trackid {0}, gating distance: {1}'.format(track.track_id, gating_distance))
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix
