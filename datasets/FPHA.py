import pickle
import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy.signal import medfilt


def get_skeleton(skeleton):
    hand_skeleton = np.empty((0, 66), float)
    for frame_id in range(0, skeleton.shape[0]):
        current_skeleton = skeleton[frame_id].reshape(21, 3)
        key_points = np.empty((0, 3), float)
        palm_array = np.empty((0, 3), float)
        for index, landmark in enumerate(current_skeleton):
            landmark_x = round(float(landmark[0]), 4)
            landmark_y = round(float(landmark[1]), 4)
            landmark_z = round(float(landmark[2]), 4)
            landmark_point = [np.array((landmark_x, landmark_y, landmark_z))]
            key_points = np.append(key_points, landmark_point, axis=0)

            if index == 0:
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 1:
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 2:
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 3:
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 4:
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 5:
                palm_array = np.append(palm_array, landmark_point, axis=0)

        motion = cv2.moments(palm_array[:, 0:2].astype(int))
        cx, cy = 0, 0
        if motion['m00'] != 0:
            cx = int(motion['m10'] / motion['m00'])
            cy = int(motion['m01'] / motion['m00'])
        cz = np.average(palm_array[:, 2])
        key_points = np.insert(key_points, [1], [round(float(cx), 4), round(float(cy), 4), round(float(cz), 4)], axis=0)
        hand_skeleton = np.append(hand_skeleton, key_points.reshape(1, 66), axis=0)

    hand_skeleton = hand_skeleton.reshape((-1, 22, 3))
    output_skeleton = hand_skeleton.copy()
    # Thumb
    output_skeleton[:, 3:6, :] = hand_skeleton[:, 7:10, :]
    # Index
    output_skeleton[:, 6, :] = hand_skeleton[:, 3, :]
    output_skeleton[:, 7:10, :] = hand_skeleton[:, 10:13, :]
    # Middle
    output_skeleton[:, 10, :] = hand_skeleton[:, 4, :]
    output_skeleton[:, 11: 14, :] = hand_skeleton[:, 13: 16, :]
    # Ring
    output_skeleton[:, 14, :] = hand_skeleton[:, 5, :]
    output_skeleton[:, 15: 18, :] = hand_skeleton[:, 16: 19, :]
    # Pinky
    output_skeleton[:, 18, :] = hand_skeleton[:, 6, :]
    output_skeleton[:, 19: 22, :] = hand_skeleton[:, 19: 22, :]

    output_skeleton = output_skeleton.reshape(-1, 66)
    return output_skeleton


def get_train_data(data_path, save_path):
    # train file path
    train_file = open(data_path+'train_gestures.txt')

    # define train dataset
    train = dict()
    train['pose'] = []
    train['label'] = []

    for line in train_file.readlines():
        line = line.strip()
        file_path = line.split(' ')[0]
        label = line.split(' ')[1]

        # define skeleton path
        skeleton_path = data_path + file_path + "/" + "skeleton.txt"
        pose = np.loadtxt(skeleton_path).astype('float32')[:, 1:]
        skeleton = get_skeleton(pose)
        # Use Median Filter
        for joint_index in range(skeleton.shape[1]):
            skeleton[:, joint_index] = medfilt(skeleton[:, joint_index])
        train['pose'].append(skeleton)
        train['label'].append(label)
    # store train data
    pickle.dump(train, open(save_path+"fpha_train.pkl", "wb"))


def get_test_data(data_path, save_path):
    # test file path
    test_file = open(data_path + 'test_gestures.txt')

    # define test dataset
    test = dict()
    test['pose'] = []
    test['label'] = []

    for line in test_file.readlines():
        line = line.strip()
        file_path = line.split(' ')[0]
        label = line.split(' ')[1]

        # define skeleton path
        skeleton_path = data_path + file_path + "/" + "skeleton.txt"
        pose = np.loadtxt(skeleton_path).astype('float32')[:, 1:]
        skeleton = get_skeleton(pose)
        # Use Median Filter
        for joint_index in range(skeleton.shape[1]):
            skeleton[:, joint_index] = medfilt(skeleton[:, joint_index])
        test['pose'].append(skeleton)
        test['label'].append(label)
    # store test data
    pickle.dump(test, open(save_path + "fpha_test.pkl", "wb"))


if __name__ == '__main__':
    data_path = "F:/dataset/Hand_pose_annotation_v1/"
    save_path = "F:/dataset/FPHA/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Successfully create save dataset folder.")
    print("Start to get train data ....")
    get_train_data(data_path, save_path)
    print("Successfully get train data.")
    print("Start to get test data ....")
    get_test_data(data_path, save_path)
    print("Successfully get test data.")

