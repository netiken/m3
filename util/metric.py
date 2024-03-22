import numpy as np
from scipy import stats


def p_error(dis_target, dis_ref, anchor=99):
    est = np.percentile(dis_target, anchor)
    gt = np.percentile(dis_ref, anchor)
    return np.round((est - gt) / gt, 3)


def p_list(dis_target, anchor_list=[70, 80, 90, 95, 99]):
    res = []
    for anchor in anchor_list:
        res.append(np.percentile(dis_target, anchor))
    return res


def p_error_list(dis_target, dis_ref, anchor_list=[80, 85, 90, 95, 99]):
    res = []
    for anchor in anchor_list:
        est = np.percentile(dis_target, anchor)
        gt = np.percentile(dis_ref, anchor)
        res.append(abs((est - gt) / gt))
    return np.round(np.sum(res) / len(anchor_list), 3)


def get_emd_distance(x, y):
    return stats.wasserstein_distance(x, y)


def get_js_distance(x, y):
    m = (x + y) / 2
    js = 0.5 * stats.entropy(x, m) + 0.5 * stats.entropy(y, m)
    return js
