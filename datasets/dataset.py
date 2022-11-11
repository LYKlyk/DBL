from tqdm import tqdm
from utils.util import *


def load_test_dataset(opt, test_sample):
    test_point = []
    test_norm = []
    test_edge = []
    test_y = []
    for i in tqdm(range(len(test_sample['pose']))):
        pose = np.copy(test_sample['pose'][i]).reshape([-1, 22, 3])
        pose = zoom(pose=pose,
                    target_length=opt.frame_length,
                    joints_num=opt.joint_number,
                    joints_dim=opt.joint_dim)
        pose = normalize_range(pose)
        if opt.class_number == 14:
            label = np.zeros(opt.class_number)
            label[test_sample['coarse_label'][i] - 1] = 1
        else:
            label = np.zeros(opt.class_number)
            label[test_sample['fine_label'][i] - 1] = 1
        edge = get_edge(pose)
        point_norm = normalize_point(pose)
        test_norm.append(point_norm)
        test_point.append(pose)
        test_edge.append(edge)
        test_y.append(label)

    test_point = np.stack(test_point)
    test_norm = np.stack(test_norm)
    test_edge = np.stack(test_edge)
    test_y = np.stack(test_y)

    return test_point, test_edge, test_norm, test_y


def load_train_dataset(opt, train_data):
    train_point = []
    train_edge = []
    train_norm = []
    train_y = []
    for i in tqdm(range(len(train_data['pose']))):
        pose = np.copy(train_data['pose'][i]).reshape([-1, 22, 3])
        pose = zoom(pose=pose,
                    target_length=opt.frame_length,
                    joints_num=opt.joint_number,
                    joints_dim=opt.joint_dim)
        pose = normalize_range(pose)
        if opt.class_number == 14:
            label = np.zeros(opt.class_number)
            label[train_data['coarse_label'][i] - 1] = 1
        else:
            label = np.zeros(opt.class_number)
            label[train_data['fine_label'][i] - 1] = 1
        edge = get_edge(pose)
        point_norm = normalize_point(pose)
        train_norm.append(point_norm)
        train_point.append(pose)
        train_edge.append(edge)
        train_y.append(label)

    train_point = np.stack(train_point)
    train_norm = np.stack(train_norm)
    train_edge = np.stack(train_edge)
    train_y = np.stack(train_y)

    return train_point, train_edge, train_norm, train_y


def load_test_fpha(opt, test_sample):
    test_point = []
    test_norm = []
    test_edge = []
    test_y = []
    for i in tqdm(range(len(test_sample['pose']))):
        pose = np.copy(test_sample['pose'][i]).reshape([-1, 22, 3])
        pose = zoom(pose=pose,
                    target_length=opt.frame_length,
                    joints_num=opt.joint_number,
                    joints_dim=opt.joint_dim)
        pose = normalize_range(pose)
        label = np.zeros(opt.class_number)
        label[int(test_sample['label'][i])] = 1
        edge = get_edge(pose)
        point_norm = normalize_point(pose)
        test_norm.append(point_norm)
        test_point.append(pose)
        test_edge.append(edge)
        test_y.append(label)

    test_point = np.stack(test_point)
    test_norm = np.stack(test_norm)
    test_edge = np.stack(test_edge)
    test_y = np.stack(test_y)

    return test_point, test_edge, test_norm, test_y


def load_train_fpha(opt, train_data):
    train_point = []
    train_edge = []
    train_norm = []
    train_y = []
    for i in tqdm(range(len(train_data['pose']))):
        pose = np.copy(train_data['pose'][i]).reshape([-1, 22, 3])
        pose = zoom(pose=pose,
                    target_length=opt.frame_length,
                    joints_num=opt.joint_number,
                    joints_dim=opt.joint_dim)
        pose = normalize_range(pose)
        label = np.zeros(opt.class_number)
        label[int(train_data['label'][i])] = 1
        edge = get_edge(pose)
        point_norm = normalize_point(pose)
        train_norm.append(point_norm)
        train_point.append(pose)
        train_edge.append(edge)
        train_y.append(label)

    train_point = np.stack(train_point)
    train_norm = np.stack(train_norm)
    train_edge = np.stack(train_edge)
    train_y = np.stack(train_y)
    return train_point, train_edge, train_norm, train_y

