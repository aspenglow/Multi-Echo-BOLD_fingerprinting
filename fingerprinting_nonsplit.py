import numpy as np
import os
import argparse
from fingerprinting.utils import *

from fingerprinting.load_data import load_data_nonsplit
from fingerprinting.pca_denoising import pca_denoising
from fingerprinting.draw_results import *

def parse_args():
    parser = argparse.ArgumentParser(description='PCA denoising with spliting timeseries to test and retest.')
    parser.add_argument('--data_path', default="./data",
                        help='path of the timeseries.')
    parser.add_argument('--result_path', default="./fingerprinting/results_nonsplit",
                        help='path to save results.')
    args = parser.parse_args()
    return args

def fingerprinting_nonsplit(data_path='./data0', result_path='./fingerprinting/results_nonsplit'):
    assert(os.path.exists(data_path))
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    echoes_total_num = 5  # including optimal combination
    subjects_total_num = int(len(os.listdir(data_path)) / (echoes_total_num)) # there is another optimal TS.
    subjects_num = subjects_total_num
    print("There are " + str(subjects_total_num) + " subjects with " + str(echoes_total_num) + " echoes (include 1 optcomb).")

    # Load data
    FCs, orig_matrixs = load_data_nonsplit(data_path, subjects_num, echoes_total_num)
    
    echo_pair_result_path = os.path.join(result_path, "echo_pair_results")
    if not os.path.exists(echo_pair_result_path):
        os.mkdir(echo_pair_result_path)
    recon_FC_root_img_path = os.path.join(result_path, "reconstructed_FCs_images")
    if not os.path.exists(recon_FC_root_img_path):
        os.mkdir(recon_FC_root_img_path)

    echo_pairs_num = int(echoes_total_num * (echoes_total_num - 1) / 2)
    self_elements_orig = np.zeros((echo_pairs_num, subjects_num), dtype=float)
    self_elements_opt = np.zeros((echo_pairs_num, subjects_num), dtype=float)
    count = 0
    ### For each echo-pair, use PCA method to get optimal principle components for matrix reconstruction.
    for echo_retest in tqdm(range(echoes_total_num), desc='echo-retest', leave=False): # Another echo_index: for optimal combination.
        for echo_test in tqdm(range(echoes_total_num), desc='echo-test', leave=False):
            if echo_test <= echo_retest:
                continue
            Ident_mat_orig, Ident_mat_recon_opt, _, _, _, _, _, _, _, _, _, _ \
                = pca_denoising(echo_test, echo_retest, orig_matrixs, orig_matrixs)
            
            mask_diag = np.diag(np.full(subjects_num, True, dtype=bool))
            self_orig = Ident_mat_orig[mask_diag]
            self_elements_orig[count] = self_orig
            self_opt = Ident_mat_recon_opt[mask_diag]
            self_elements_opt[count] = self_opt
            count += 1
        
    # Save self identifiabilities before denoising and after denoising. 
    echo_pair_violin_path = os.path.join(result_path, "echo_pair_violin")
    if not os.path.exists(echo_pair_violin_path):
        os.mkdir(echo_pair_violin_path)

    draw_echo_pairs_self_violin(echo_pair_violin_path, self_elements_orig, self_elements_opt)

   
if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    result_path = args.result_path

    fingerprinting_nonsplit(data_path, result_path)