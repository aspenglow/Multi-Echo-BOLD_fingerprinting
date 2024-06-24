import numpy as np
import os
import argparse
from fingerprinting.utils import *

from fingerprinting.draw_results import draw_Idiff_Iself_Iothers

def parse_args():
    parser = argparse.ArgumentParser(description='PCA denoising with spliting timeseries to test and retest.')
    parser.add_argument('--data_path', default="./fingerprinting/results/Idiff_root",
                        help='path to save results.')
    parser.add_argument('--result_path', default="./fingerprinting/results_split",
                        help='path to save results.')
    args = parser.parse_args()
    return args

def calculate_bootstraping_result(data_path, result_path):
    Idiff_orig_path = os.path.join(data_path, "Idiff_orig_txt")
    Iself_orig_path = os.path.join(data_path, "Iself_orig_txt")
    Iothers_orig_path = os.path.join(data_path, "Iothers_orig_txt")
    Idiff_opt_path = os.path.join(data_path, "Idiff_opt_txt")
    Iself_opt_path = os.path.join(data_path, "Iself_opt_txt")
    Iothers_opt_path = os.path.join(data_path, "Iothers_opt_txt")
    assert(os.path.exists(Idiff_orig_path))
    assert(os.path.exists(Idiff_opt_path))
    assert(os.path.exists(Iself_orig_path))
    assert(os.path.exists(Iself_opt_path))
    assert(os.path.exists(Iothers_orig_path))
    assert(os.path.exists(Iothers_opt_path))

    bootstrap_num = len(os.listdir(Iothers_opt_path))
    print("number of bootstraps: " + str(bootstrap_num))
    Idiff_orig_list = os.listdir(Idiff_orig_path)
    Idiff_orig_list.sort()
    Idiff_opt_list = os.listdir(Idiff_opt_path)
    Idiff_opt_list.sort()

    Iself_orig_list = os.listdir(Iself_orig_path)
    Iself_orig_list.sort()
    Iself_opt_list = os.listdir(Iself_opt_path)
    Iself_opt_list.sort()

    Iothers_orig_list = os.listdir(Iothers_orig_path)
    Iothers_orig_list.sort()
    Iothers_opt_list = os.listdir(Iothers_opt_path)
    Iothers_opt_list.sort()

    test_data = np.genfromtxt(fname=os.path.join(Idiff_orig_path, Idiff_orig_list[0]), dtype='float32', delimiter=' ')
    echoes_total_num = test_data.shape[0]
    
    Idiff_mat_orig = np.zeros((echoes_total_num, echoes_total_num), dtype=float)
    Idiff_mat_opt = np.zeros((echoes_total_num, echoes_total_num), dtype=float)
    Iself_mat_orig = np.zeros((echoes_total_num, echoes_total_num), dtype=float)
    Iself_mat_opt = np.zeros((echoes_total_num, echoes_total_num), dtype=float)
    Iothers_mat_orig = np.zeros((echoes_total_num, echoes_total_num), dtype=float)
    Iothers_mat_opt = np.zeros((echoes_total_num, echoes_total_num), dtype=float)
    
    for file in Idiff_orig_list:
        Idiff_mat_orig += np.genfromtxt(fname=os.path.join(Idiff_orig_path, file), dtype='float32', delimiter=' ')
    Idiff_mat_orig /= bootstrap_num

    for file in Idiff_opt_list:
        Idiff_mat_opt += np.genfromtxt(fname=os.path.join(Idiff_opt_path, file), dtype='float32', delimiter=' ')
    Idiff_mat_opt /= bootstrap_num
    
    for file in Iself_orig_list:
        Iself_mat_orig += np.genfromtxt(fname=os.path.join(Iself_orig_path, file), dtype='float32', delimiter=' ')
    Iself_mat_orig /= bootstrap_num

    for file in Iself_opt_list:
        Iself_mat_opt += np.genfromtxt(fname=os.path.join(Iself_opt_path, file), dtype='float32', delimiter=' ')
    Iself_mat_opt /= bootstrap_num

    for file in Iothers_orig_list:
        Iothers_mat_orig += np.genfromtxt(fname=os.path.join(Iothers_orig_path, file), dtype='float32', delimiter=' ')
    Iothers_mat_orig /= bootstrap_num

    for file in Iothers_opt_list:
        Iothers_mat_opt += np.genfromtxt(fname=os.path.join(Iothers_opt_path, file), dtype='float32', delimiter=' ')
    Iothers_mat_opt /= bootstrap_num

    image_path = os.path.join(result_path, "bootstrap_result_" + str(bootstrap_num))
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    draw_Idiff_Iself_Iothers(image_path, Idiff_mat_orig, Idiff_mat_opt, Iself_mat_orig, Iself_mat_opt, Iothers_mat_orig, Iothers_mat_opt)
    save_Idiff_Iself_Iothers_txt(image_path, Idiff_mat_orig, Idiff_mat_opt, Iself_mat_orig, Iself_mat_opt, Iothers_mat_orig, Iothers_mat_opt)

if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    result_path = args.result_path
    calculate_bootstraping_result(data_path, result_path)
