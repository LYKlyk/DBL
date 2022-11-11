import numpy as np
import scipy.ndimage.interpolation as inter
from scipy.signal import medfilt


def zoom(pose, target_length=64, joints_num=25, joints_dim=3):
    l = pose.shape[0]
    p_new = np.empty([target_length, joints_num, joints_dim])
    for m in range(joints_num):
        for n in range(joints_dim):
            p_new[:, m, n] = medfilt(p_new[:, m, n], 3)
            p_new[:, m, n] = inter.zoom(pose[:, m, n], target_length / l)[:target_length]
    return p_new


def normalize_range(p):
    p[:, :, 0] = p[:, :, 0] - np.mean(p[:, :, 0])
    p[:, :, 1] = p[:, :, 1] - np.mean(p[:, :, 1])
    p[:, :, 2] = p[:, :, 2] - np.mean(p[:, :, 2])
    return p


def normalize_point(p):
    pose = p
    pose = np.insert(pose, 0, values=pose[:, 1, :], axis=1)
    pose = np.insert(pose, -1, values=pose[:, 1, :], axis=1)
    pose[:, :, 0] = (pose[:, :, 0] - np.min(pose[:, :, 0]))/(np.max(pose[:, :, 0]) - np.min(pose[:, :, 0]))
    pose[:, :, 1] = (pose[:, :, 1] - np.min(pose[:, :, 1]))/(np.max(pose[:, :, 1]) - np.min(pose[:, :, 1]))
    pose[:, :, 2] = (pose[:, :, 2] - np.min(pose[:, :, 2]))/(np.max(pose[:, :, 2]) - np.min(pose[:, :, 2]))
    return pose


def get_edge(pose):
    pose_edge = []
    # deal with wrist
    edge = pose[:, 1, :] - pose[:, 0, :]
    pose_edge.append(edge)
    # deal with thumb finger
    edge = pose[:, 1, :] - pose[:, 2, :]
    pose_edge.append(edge)
    for index in range(2, 5):
        edge = pose[:, index, :] - pose[:, index+1, :]
        pose_edge.append(edge)
    # deal with index finger
    edge = pose[:, 1, :] - pose[:, 6, :]
    pose_edge.append(edge)
    for index in range(6, 9):
        edge = pose[:, index, :] - pose[:, index+1, :]
        pose_edge.append(edge)
    # deal with middle finger
    edge = pose[:, 1, :] - pose[:, 10, :]
    pose_edge.append(edge)
    for index in range(10, 13):
        edge = pose[:, index, :] - pose[:, index+1, :]
        pose_edge.append(edge)
    # deal with ring finger
    edge = pose[:, 1, :] - pose[:, 14, :]
    pose_edge.append(edge)
    for index in range(14, 17):
        edge = pose[:, index, :] - pose[:, index+1, :]
        pose_edge.append(edge)
    # deal with little finger
    edge = pose[:, 1, :] - pose[:, 18, :]
    pose_edge.append(edge)
    for index in range(18, 21):
        edge = pose[:, index, :] - pose[:, index+1, :]
        pose_edge.append(edge)
    # deal with all finger
    pose_edge = np.stack(pose_edge)
    pose_edge = np.transpose(pose_edge, (1, 0, 2))
    return pose_edge