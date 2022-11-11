from tqdm import tqdm
import pickle
import os


def get_train_test_data(valid_data, save_path, test_subject_id):
    # define train dataset
    train = dict()
    train['pose'] = []
    train['coarse_label'] = []
    train['fine_label'] = []

    # define test dataset
    test = dict()
    test['pose'] = []
    test['coarse_label'] = []
    test['fine_label'] = []

    # foreach dhg dataset
    for gesture_id in tqdm(range(1, 15)):
        for finger_id in range(1, 3):
            for subject_id in range(1, 21):
                for essai_id in range(1, 6):
                    # set dataset's path
                    key = "{}_{}_{}_{}".format(gesture_id, finger_id, subject_id, essai_id)
                    # set use to coarse gesture
                    coarse_label = gesture_id
                    if finger_id == 1:
                        fine_label = gesture_id
                    else:
                        fine_label = gesture_id+14
                    # add sample data into validate data list
                    if subject_id == test_subject_id:
                        test['pose'].append(valid_data[key])
                        test['coarse_label'].append(coarse_label)
                        test['fine_label'].append(fine_label)
                    else:
                        train['pose'].append(valid_data[key])
                        train['coarse_label'].append(coarse_label)
                        train['fine_label'].append(fine_label)
    # print error information
    if len(test) == 0:
        print("No validate dataset.")
    print("Successfully establish train and validate dataset.")
    # store train data
    pickle.dump(train, open(save_path + "/dhg_train_"+str(test_subject_id)+".pkl", "wb"))
    # store test data
    pickle.dump(test, open(save_path+"/dhg_test_"+str(test_subject_id)+".pkl", "wb"))


def parse_dhg_datafile(data_file):
    video = []
    for line in data_file:
        line = line.split("\n")[0]
        data = line.split(" ")
        frame = []
        point = []
        for element in data:
            # add hand's skeleton point location (x,y,z)
            point.append(float(element))
            if len(point) == 3:
                # add hand's every skeleton point
                frame.append(point)
                point = []
        # add every frame's all skeletons into video
        video.append(frame)
    # return video
    return video


def read_dhg_dataset(root_path):
    # define dhg dataset result
    dhg_result = {}
    print("Start to deal with DHG dataset.")
    # foreach dhg dataset
    for gesture_id in range(1, 15):
        print("Begin deal with folder: ", root_path+"/"+str(gesture_id))
        for finger_id in range(1, 3):
            for subject_id in range(1, 21):
                for essai_id in range(1, 6):
                    src_path = root_path + "/gesture_{}/finger_{}/subject_{}/essai_{}/skeleton_world.txt".\
                        format(gesture_id, finger_id, subject_id, essai_id)
                    with open(src_path, 'r') as src_file:
                        # get hand skeleton information
                        skeleton = parse_dhg_datafile(src_file)
                        # get store location
                        key = "{}_{}_{}_{}".format(gesture_id, finger_id, subject_id, essai_id)
                        dhg_result[key] = skeleton
    print("Successfully deal with DHG dataset.")
    # return dhg dataset result
    return dhg_result


def get_valid_frame(root_path, video_data):
    annotations_path = root_path + "/informations_troncage_sequences.txt"
    with open(annotations_path, 'r') as annotations_file:
        for line in annotations_file:
            line = line.split("\n")[0]
            data = line.split(" ")
            gesture_id = data[0]
            finger_id = data[1]
            subject_id = data[2]
            essai_id = data[3]
            # get data path
            key = "{}_{}_{}_{}".format(gesture_id, finger_id, subject_id, essai_id)
            # get start frame
            start_frame = int(data[4])
            # get end frame
            end_frame = int(data[5])
            data = video_data[key]
            video_data[key] = data[start_frame: end_frame+1]
    return video_data


def main(root_path, save_path, test_subject_id):
    # get video data
    video_data = read_dhg_dataset(root_path)
    # get dhg data valid frame
    valid_data = get_valid_frame(root_path, video_data)
    # get train and test data
    get_train_test_data(valid_data, save_path, test_subject_id)


if __name__ == '__main__':
    data_folder_path = "E:/liyangke/DHG2016"
    save_folder_path = "E:/liyangke/DHG"
    test_subject = 3
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
        print("Successfully create save dataset folder.")
    print("Start to get train and test data ....")
    main(data_folder_path, save_folder_path, test_subject)
    print("Successfully get train and test data.")
