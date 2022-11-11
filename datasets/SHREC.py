import numpy as np
from tqdm import tqdm
from scipy.signal import medfilt
import pickle
import os


def get_train_data(data_path, save_path):
    # define train list
    train_list = np.loadtxt(data_path+'train_gestures.txt').astype('int16')
    # define train dataset
    train = dict()
    train['pose'] = []
    train['coarse_label'] = []
    train['fine_label'] = []

    for frame_index in tqdm(range(len(train_list))):
        gesture_index = train_list[frame_index][0]
        finger_index = train_list[frame_index][1]
        subject_index = train_list[frame_index][2]
        essai_index = train_list[frame_index][3]
        coarse_label = train_list[frame_index][4]
        fine_label = train_list[frame_index][5]
        # define skeleton path
        skeleton_path = data_path + '/gesture_' + str(gesture_index) + '/finger_' + str(finger_index) +\
                        '/subject_' + str(subject_index) + '/essai_' + str(essai_index) + '/'
        skeleton = np.loadtxt(skeleton_path+"skeletons_world.txt").astype('float32')
        for joint_index in range(skeleton.shape[1]):
            skeleton[:, joint_index] = medfilt(skeleton[:, joint_index])
        train['pose'].append(skeleton)
        train['coarse_label'].append(coarse_label)
        train['fine_label'].append(fine_label)
    # store train data
    pickle.dump(train, open(save_path+"shrec_train.pkl", "wb"))


def get_test_data(data_path, save_path):
    # define test list
    test_list = np.loadtxt(data_path+'test_gestures.txt').astype('int16')
    # define test dataset
    test = dict()
    test['pose'] = []
    test['coarse_label'] = []
    test['fine_label'] = []

    for frame_index in tqdm(range(len(test_list))):
        gesture_index = test_list[frame_index][0]
        finger_index = test_list[frame_index][1]
        subject_index = test_list[frame_index][2]
        essai_index = test_list[frame_index][3]
        coarse_label = test_list[frame_index][4]
        fine_label = test_list[frame_index][5]
        # define skeleton path
        skeleton_path = data_path + '/gesture_' + str(gesture_index) + '/finger_' + str(finger_index) + \
                        '/subject_' + str(subject_index) + '/essai_' + str(essai_index) + '/'
        skeleton = np.loadtxt(skeleton_path + "skeletons_world.txt").astype('float32')
        for joint_index in range(skeleton.shape[1]):
            skeleton[:, joint_index] = medfilt(skeleton[:, joint_index])
        test['pose'].append(skeleton)
        test['coarse_label'].append(coarse_label)
        test['fine_label'].append(fine_label)
    # store test data
    pickle.dump(test, open(save_path+"shrec_test.pkl", "wb"))


if __name__ == '__main__':
    data_path = "F:/dataset/SHREC2017/"
    save_path = "F:/dataset/SHREC/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Successfully create save dataset folder.")
    print("Start to get train data ....")
    get_train_data(data_path, save_path)
    print("Successfully get train data.")
    print("Start to get test data ....")
    get_test_data(data_path, save_path)
    print("Successfully get test data.")

