import os
import pickle
from models.network import *
from datasets.dataset import *
from tensorflow.keras.models import load_model
from utils.options import *
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def main():
    opt = parse_options()

    # get test data
    test_data = pickle.load(open(opt.data_path + opt.test_file, "rb"))

    if opt.dataset == "FPHA":
        test_point, test_edge, test_norm, test_y = load_test_fpha(opt, test_data)
    else:
        test_point, test_edge, test_norm, test_y = load_test_dataset(opt, test_data)

    # load model
    model = load_model(opt.store_path+"/"+opt.checkpoint_weight,
                       custom_objects={'short_pose_difference': short_pose_difference})

    _, val_result_loss, _, _, _, val_result_acc, _, _, _ = model.evaluate([test_point, test_edge, test_norm],
                                                                          [test_y, test_y, test_y, test_norm],
                                                                          batch_size=opt.batch_size)
    print('val_result_loss', val_result_loss)
    print('accuracy', val_result_acc)

    start = time.time()
    pred_result, _, _, _ = model.predict([test_point, test_edge, test_norm])
    end = time.time()
    total_time = end - start
    # The total number of samples is 840, and each sample contains 32 frames
    print("FPS: ", 1.0/(total_time/(840*32)))
    
if __name__ == '__main__':
    main()

