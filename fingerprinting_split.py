import numpy as np
import os
import argparse
from fingerprinting.utils import *

from fingerprinting.load_data import load_data_split
from fingerprinting.pca_denoising import pca_denoising
from fingerprinting.draw_results import *
from fingerprinting.ICC import calculate_ICC

def parse_args():
    parser = argparse.ArgumentParser(description='PCA denoising with spliting timeseries to test and retest.')
    parser.add_argument('--data_path', default="./data0",
                        help='path of the timeseries.')
    parser.add_argument('--result_path', default="./results/results_split",
                        help='path to save results.')
    parser.add_argument('--subjects_num', type=int, default=-1, 
                        help='number of subjects to calculate.')
    parser.add_argument('--bootstrap_num', type=int, default=1, 
                        help='number of boottraps and then calculate average.')
    parser.add_argument('--icc', type=bool, default=True,
                        help='if calculate ICC of Functional connectome.')
    parser.add_argument('--save_FC', type=bool, default=False,
                        help='if save calculated FCs.')
    args = parser.parse_args()
    return args

def fingerprinting_split(data_path='./data0', result_path='./fingerprinting/results_split', subjects_num=-1, if_icc=False, if_save_FC=False):
    assert(os.path.exists(data_path))
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    echoes_total_num = 5  # including optimal combination
    echo_optcomb = echoes_total_num - 1 # the last echo is the optcomb
    subjects_total_num = int(len(os.listdir(data_path)) / (echoes_total_num)) # there is another optimal TS.
    if subjects_num == -1:
        subjects_num = subjects_total_num
    print("There are " + str(subjects_total_num) + " subjects with " + str(echoes_total_num) + " echoes (include 1 optcomb).")

    # Load data
    TSs_test, TSs_retest, FCs_test, FCs_retest, orig_matrixs_test, orig_matrixs_retest = load_data_split(data_path, subjects_num, echoes_total_num)
    
    echo_pair_result_path = os.path.join(result_path, "echo_pair_results")
    if not os.path.exists(echo_pair_result_path):
        os.mkdir(echo_pair_result_path)
    recon_FC_root_img_path = os.path.join(result_path, "reconstructed_FCs_images")
    if not os.path.exists(recon_FC_root_img_path):
        os.mkdir(recon_FC_root_img_path)

    Idiff_mat_orig = np.zeros((echoes_total_num, echoes_total_num), dtype=float)
    Idiff_mat_opt = np.zeros((echoes_total_num, echoes_total_num), dtype=float)
    Iself_mat_orig = np.zeros((echoes_total_num, echoes_total_num), dtype=float)
    Iself_mat_opt = np.zeros((echoes_total_num, echoes_total_num), dtype=float)
    Iothers_mat_orig = np.zeros((echoes_total_num, echoes_total_num), dtype=float)
    Iothers_mat_opt = np.zeros((echoes_total_num, echoes_total_num), dtype=float)
    
    if if_icc or if_save_FC:
        FCs_test_recon = np.zeros(FCs_test.shape, dtype=float)
        FCs_retest_recon = np.zeros(FCs_retest.shape, dtype=float)

    ### For each echo-pair, use PCA method to get optimal principle components for matrix reconstruction.
    for echo_test in tqdm(range(echoes_total_num), desc='echo1', leave=False): # Another echo_index: for optimal combination.
        for echo_retest in tqdm(range(echoes_total_num), desc='echo2', leave=False):
            Ident_mat_orig, Ident_mat_recon_opt, recon_matrix_opt_test, recon_matrix_opt_retest, PCA_comps_range, \
            Idiff_orig, Idiff_recon, Idiff_opt, Iself_orig, Iself_opt, Iothers_orig, Iothers_opt \
                = pca_denoising(echo_test, echo_retest, orig_matrixs_test, orig_matrixs_retest)
            
            Idiff_mat_orig[echo_retest, echo_test] = Idiff_orig 
            Iself_mat_orig[echo_retest, echo_test] = Iself_orig 
            Iothers_mat_orig[echo_retest, echo_test] = Iothers_orig 
            Idiff_mat_opt[echo_retest, echo_test] = Idiff_opt 
            Iself_mat_opt[echo_retest, echo_test] = Iself_opt 
            Iothers_mat_opt[echo_retest, echo_test] = Iothers_opt  

            draw_echo_pair_results(echo_pair_result_path, Ident_mat_orig, Ident_mat_recon_opt, PCA_comps_range, 
                    Idiff_orig, Idiff_recon, Idiff_opt, echo_test, echo_retest, echo_optcomb) 

            if if_icc or if_save_FC:
                FC_test_recon, FC_retest_recon = FC_reconstruct(recon_FC_root_img_path, echo_test, echo_retest, echoes_total_num, \
                        recon_matrix_opt_test, recon_matrix_opt_retest, FCs_test, FCs_retest, if_save_FC)
                if echo_test == echo_retest:
                    FCs_test_recon[echo_test] = FC_test_recon
                    FCs_retest_recon[echo_retest] = FC_retest_recon
                
        
    # Save Idiff Iself Iothers echo-pair matrix before denoising and after denoising. 
    Idiff_root_path = os.path.join(result_path, "Idiff_root")
    if not os.path.exists(Idiff_root_path):
        os.mkdir(Idiff_root_path)

    Idiff_image_path = os.path.join(Idiff_root_path, "Idiff_Iself_Iothers_images")
    if not os.path.exists(Idiff_image_path):
        os.mkdir(Idiff_image_path)

    draw_Idiff_Iself_Iothers(Idiff_image_path, Idiff_mat_orig, Idiff_mat_opt, Iself_mat_orig, Iself_mat_opt, Iothers_mat_orig, Iothers_mat_opt)
    save_Idiff_Iself_Iothers_txt(Idiff_root_path, Idiff_mat_orig, Idiff_mat_opt, Iself_mat_orig, Iself_mat_opt, Iothers_mat_orig, Iothers_mat_opt)

    if if_icc:
        # Compute edgewise ICC with original and optimal reconstruction.
        ICC_path = os.path.join(result_path, "ICC")
        if not os.path.exists(ICC_path):
            os.mkdir(ICC_path)
        ICCs = calculate_ICC(FCs_test, FCs_retest)
        ICCs_recon = calculate_ICC(FCs_test_recon, FCs_retest_recon)
        for echo_index in range(echoes_total_num):
            draw_ICC(ICC_path, ICCs, ICCs_recon, echo_index, echo_optcomb)

if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    result_path = args.result_path
    subjects_num = args.subjects_num
    bootstrap_num = args.bootstrap_num
    if_icc = args.icc
    if_save_FC = args.save_FC
    for i in tqdm(range(bootstrap_num), desc='bootstraping count', leave=False):
        fingerprinting_split(data_path, result_path, subjects_num, if_icc, if_save_FC)