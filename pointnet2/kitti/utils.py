import numpy as np

def transform_se3(pts, tf):
    assert(pts.shape[1] == 3)
    pts_1 = np.concatenate((pts, np.ones((pts.shape[0],1))), axis=1)
    pts_tf_1 = tf.dot(pts_1.T).T
    pts_tf = pts_tf_1[:,0:3]
    return pts_tf


def read_calibration(path):
    calib = {}
    with open(path) as f:
        for line in f:
            fields = line.split()
            if len(fields) == 0: continue
            keyword = fields[0].strip(':')
            calib[keyword] = np.array([float(x) for x in fields[1:]]).reshape(3, -1)
    velo_to_cam_tf = np.concatenate((calib['Tr_velo_to_cam'], np.array([[0,0,0,1]])),axis=0)
    return velo_to_cam_tf


def read_detection(path):
    det = []
    with open(path) as f:
        for line in f:
            elements = line.split()
            if len(elements) == 15:
                cls, trunc, occl, alpha, x1, y1, x2, y2, h, w, l, cx, cy, cz, ry = elements
                score = float('nan')
            else:
                cls, trunc, occl, alpha, x1, y1, x2, y2, h, w, l, cx, cy, cz, ry, score = elements

            if cls=='DontCare': continue
            det.append([cls, float(h), float(w), float(l), float(cx), float(cy), float(cz), float(ry), float(score)])

    return det


# we do per point NMS with score
# to make sure segments we produce do not overlap
def convert_dets_to_segs(pts_velo_cs, velo_to_cam_tf, dets):
    pts_cam_cs = transform_se3(pts_velo_cs[:,:3], velo_to_cam_tf)

    point_labels = np.full(len(pts_velo_cs), -1)
    point_scores = np.full(len(pts_velo_cs), -10000000.0)
    for i in range(len(dets)):
        cls, height, width, length, cx, cy, cz, ry, score = dets[i]
        ry = ry + np.pi/2
        obj_to_cam_tf = np.array([[ np.cos(ry), 0, np.sin(ry), cx],
                                  [          0, 1,          0, cy],
                                  [-np.sin(ry), 0, np.cos(ry), cz],
                                  [          0, 0,          0,  1]])
        cam_to_obj_tf = np.linalg.inv(obj_to_cam_tf)
        pts_obj_cs = transform_se3(pts_cam_cs, cam_to_obj_tf)
        test_x = np.logical_and(pts_obj_cs.T[0] >= -width/2.0,
                                pts_obj_cs.T[0] <=  width/2.0)
        test_y = np.logical_and(pts_obj_cs.T[1] >=    -height,
                                pts_obj_cs.T[1] <=          0)
        test_z = np.logical_and(pts_obj_cs.T[2] >= -length/2.0,
                                pts_obj_cs.T[2] <=  length/2.0)
        # a binary mask that indicates if points are inside the bounding box
        space_mask = np.logical_and(test_x, np.logical_and(test_y, test_z))
        # a binary mask that indicates if the bounding box has a higher score
        score_mask = score > point_scores
        #
        final_mask = np.logical_and(space_mask, score_mask)
        point_labels[final_mask] = i
        point_scores[final_mask] = score

    segs = []
    scores = []
    classes = []
    for i in range(len(dets)):
        I = np.flatnonzero(point_labels == i)
        if len(I) > 0:
            segs.append(I)
            scores.append(dets[i][-1])
            classes.append(dets[i][0])
    # return segs, scores
    return segs, scores, classes


# we do not do per point NMS with score
# we allow overlap between ground truth segments
# because we will take care of it during evaluation
def convert_gtdets_to_gtsegs(pts_velo_cs, velo_to_cam_tf, gtdets):
    pts_cam_cs = transform_se3(pts_velo_cs[:,:3], velo_to_cam_tf)

    gtsegs = []
    gtclasses = []
    for i in range(len(gtdets)):
        cls, height, width, length, cx, cy, cz, ry, _ = gtdets[i]
        ry = ry + np.pi/2
        obj_to_cam_tf = np.array([[ np.cos(ry), 0, np.sin(ry), cx],
                                  [          0, 1,          0, cy],
                                  [-np.sin(ry), 0, np.cos(ry), cz],
                                  [          0, 0,          0,  1]])
        cam_to_obj_tf = np.linalg.inv(obj_to_cam_tf)
        pts_obj_cs = transform_se3(pts_cam_cs, cam_to_obj_tf)
        test_x = np.logical_and(pts_obj_cs.T[0] >= -width/2.0,
                                pts_obj_cs.T[0] <=  width/2.0)
        test_y = np.logical_and(pts_obj_cs.T[1] >=    -height,
                                pts_obj_cs.T[1] <=          0)
        test_z = np.logical_and(pts_obj_cs.T[2] >= -length/2.0,
                                pts_obj_cs.T[2] <=  length/2.0)
        # a binary mask that indicates if points are inside the bounding box
        space_mask = np.logical_and(test_x, np.logical_and(test_y, test_z))
        #
        gtsegs.append(np.flatnonzero(space_mask))
        gtclasses.append(cls)

    return gtsegs, gtclasses
