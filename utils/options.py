import argparse


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="F:/dataset/SHREC/", type=str,
                        help="Root directory of dataset.")
    parser.add_argument("--store_path", default="./shrec/14/store", type=str,
                        help="The path to store model.")
    parser.add_argument("--test_file", default="shrec_test.pkl", type=str,
                        help="The name of test data file.")
    parser.add_argument("--checkpoint_weight", default="DBL-SHREC-14.h5", type=str,
                        help="Trained model weight.")

    parser.add_argument("--dataset", default="SHREC", type=str, choices=["SHREC", "FPHA", "DHG"],
                        help="Current used dataset.")
    parser.add_argument("--joint_number", default=22, type=int,
                        help="The number of hand joint.")
    parser.add_argument("--joint_dim", default=3, type=int,
                        help="The number of hand joint dimension.")
    parser.add_argument("--class_number", default=14, type=int, choices=[14, 28, 45],
                        help="The number of labels.")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="The size of batch.")
    parser.add_argument("--frame_length", default=32, type=int,
                        help="The length of sample time.")

    opt = parser.parse_args()
    return opt


