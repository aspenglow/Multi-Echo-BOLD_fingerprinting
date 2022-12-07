import numpy as np
import scipy.stats as stats
from tqdm import tqdm
import time
import os

from fingerprinting.draw_results import draw_FC_reconstruction

def FC_reconstruct(recon_FC_root_img_path, echo_test, echo_retest, echoes_total_num, recon_matrix_opt_test, recon_matrix_opt_retest, FCs_test, FCs_retest, if_save_FC):
    subjects_num = recon_matrix_opt_test.shape[1]
    FC_side_length = FCs_test[0,0].shape[0]
    mask = np.tril(np.full((FC_side_length, FC_side_length), True, dtype=bool), -1) 
    FCs_test_recon = np.zeros(FCs_test.shape[1:])
    FCs_retest_recon = np.zeros(FCs_retest.shape[1:])
    # Reconstruct FC from recon_matrix
    # Each FC will have overlaps with # of echoes. We consider to average those overlaps. 
    for subject_index in tqdm(range(subjects_num), desc='subject', leave=False):
        FC_test_recon = np.identity(FC_side_length, dtype=float)
        FC_test_recon[mask] = recon_matrix_opt_test[:,subject_index]
        FC_test_recon = FC_test_recon.transpose()
        FC_test_recon[mask] = recon_matrix_opt_test[:,subject_index] 
        
        FC_retest_recon = np.identity(FC_side_length, dtype=float)
        FC_retest_recon[mask] = recon_matrix_opt_retest[:,subject_index]
        FC_retest_recon = FC_retest_recon.transpose()
        FC_retest_recon[mask] = recon_matrix_opt_retest[:,subject_index]
        
        FCs_test_recon[subject_index] = FC_test_recon
        FCs_retest_recon[subject_index] = FC_retest_recon
        
        if if_save_FC:
            # Save reconstructed FC by each echo pair.
            FC_test_orig = FCs_test[echo_test, subject_index]
            FC_retest_orig = FCs_retest[echo_retest, subject_index]
            draw_FC_reconstruction(recon_FC_root_img_path, FC_test_orig, FC_test_recon, FC_retest_orig, FC_retest_recon, subject_index, echo_test, echo_retest, echo_optcomb=echoes_total_num-1)
        
        return FCs_test_recon, FCs_retest_recon

def save_Idiff_Iself_Iothers_txt(Idiff_root_path, Idiff_mat_orig, Idiff_mat_opt, Iself_mat_orig, Iself_mat_opt, Iothers_mat_orig, Iothers_mat_opt):
    Idiff_orig_path = os.path.join(Idiff_root_path, "Idiff_orig_txt")
    if not os.path.exists(Idiff_orig_path):
        os.mkdir(Idiff_orig_path)

    Iself_orig_path = os.path.join(Idiff_root_path, "Iself_orig_txt")
    if not os.path.exists(Iself_orig_path):
        os.mkdir(Iself_orig_path)

    Iothers_orig_path = os.path.join(Idiff_root_path, "Iothers_orig_txt")
    if not os.path.exists(Iothers_orig_path):
        os.mkdir(Iothers_orig_path)

    Idiff_opt_path = os.path.join(Idiff_root_path, "Idiff_opt_txt")
    if not os.path.exists(Idiff_opt_path):
        os.mkdir(Idiff_opt_path)

    Iself_opt_path = os.path.join(Idiff_root_path, "Iself_opt_txt")
    if not os.path.exists(Iself_opt_path):
        os.mkdir(Iself_opt_path)

    Iothers_opt_path = os.path.join(Idiff_root_path, "Iothers_opt_txt")
    if not os.path.exists(Iothers_opt_path):
        os.mkdir(Iothers_opt_path)

    timestamp = int(time.time())
    current_time = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(timestamp))
    np.savetxt(os.path.join(Idiff_orig_path, "Idiff_orig_" + current_time + ".txt"), Idiff_mat_orig)
    np.savetxt(os.path.join(Idiff_opt_path, "Idiff_opt_" + current_time + ".txt"), Idiff_mat_opt)
    np.savetxt(os.path.join(Iself_orig_path, "Iself_orig_" + current_time + ".txt"), Iself_mat_orig)
    np.savetxt(os.path.join(Iself_opt_path, "Iself_opt_" + current_time + ".txt"), Iself_mat_opt)
    np.savetxt(os.path.join(Iothers_orig_path, "Iothers_orig_" + current_time + ".txt"), Iothers_mat_orig)
    np.savetxt(os.path.join(Iothers_opt_path, "Iothers_opt_" + current_time + ".txt"), Iothers_mat_opt)
    