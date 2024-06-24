import numpy as np
import os
import argparse

from fingerprinting.load_data import load_data_split
from fingerprinting.pca_denoising import pca_denoising
from fingerprinting.draw_results import *
from fingerprinting.ICC import calculate_ICC
from fingerprinting.utils import *
from fingerprinting.utils_identifiability import *

def parse_args(): 
    parser = argparse.ArgumentParser(description='PCA denoising with spliting timeseries to test and retest.')
    parser.add_argument('--data_path', default="./data0",
                        help='path of the timeseries.')
    parser.add_argument('--result_path', default="./results/results_speedup",
                        help='path to save results.')
    parser.add_argument('--subjects_num', type=int, default=-1, 
                        help='number of subjects to calculate.')
    parser.add_argument('--bootstrap_num', type=int, default=1, 
                        help='number of boottraps and then calculate average.')
    parser.add_argument('--parallel', type=bool, default=False,
                        help='if compute PCA denoising in parallel.')
    parser.add_argument('--icc', type=bool, default=False,
                        help='if calculate ICC of Functional connectome.')
    parser.add_argument('--save_FC', type=bool, default=False,
                        help='if save calculated FCs.')
    args = parser.parse_args()
    return args

def fingerprinting_speedup(data_path='./data0', result_path='./results/results_speedup', subjects_num=-1, if_icc=False, if_save_FC=False):
    assert(os.path.exists(data_path))
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    echo_pair_result_path = os.path.join(result_path, "echo_pair_results")
    if not os.path.exists(echo_pair_result_path):
        os.mkdir(echo_pair_result_path)
    recon_FC_root_img_path = os.path.join(result_path, "reconstructed_FCs_images")
    if not os.path.exists(recon_FC_root_img_path):
        os.mkdir(recon_FC_root_img_path)
    ICC_path = os.path.join(result_path, "ICC")
    if not os.path.exists(ICC_path):
        os.mkdir(ICC_path)

    echoes_total_num = 5  # including optimal combination
    echo_optcomb = echoes_total_num - 1 # the last echo is the optcomb
    subjects_total_num = int(len(os.listdir(data_path)) / (echoes_total_num)) # there is another optimal TS.
    if subjects_num == -1:
        subjects_num = subjects_total_num
    print("There are " + str(subjects_total_num) + " subjects with " + str(echoes_total_num) + " echoes (include 1 optcomb).")

    # Load data
    TCs_test, TCs_retest, FCs_test, FCs_retest, orig_matrixs_test, orig_matrixs_retest = load_data_split(data_path, subjects_num, echoes_total_num)
    TCs = np.concatenate((TCs_test, TCs_retest), axis=3)

    tSNRs = compute_tSNR_from_TCs(TCs)
    draw_echoes_tSNRs(result_path, tSNRs)

    t_FCs = np.zeros((orig_matrixs_test.shape[0],orig_matrixs_test.shape[2],orig_matrixs_test.shape[1]))
    for echo in range(TCs.shape[0]):
        # t_FCs[echo] = FCs_normalize(compute_FCs(TCs[echo]))
        t_FCs[echo] = compute_FCs(TCs[echo])
    t_FCs = t_FCs[:,3]
    draw_compare_links_with_different_echoes(result_path, t_FCs)
    

    Idiff_mat_orig = np.zeros((echoes_total_num, echoes_total_num), dtype=float)
    Idiff_mat_opt = np.zeros((echoes_total_num, echoes_total_num), dtype=float)
    Iself_mat_orig = np.zeros((echoes_total_num, echoes_total_num), dtype=float)
    Iself_mat_opt = np.zeros((echoes_total_num, echoes_total_num), dtype=float)
    Iothers_mat_orig = np.zeros((echoes_total_num, echoes_total_num), dtype=float)
    Iothers_mat_opt = np.zeros((echoes_total_num, echoes_total_num), dtype=float)
    
    # if if_icc or if_save_FC:
    #     FCs_test_recon = np.zeros(FCs_test.shape, dtype=float)
    #     FCs_retest_recon = np.zeros(FCs_retest.shape, dtype=float)

    yeoOrder,limit_yeo = load_yeonets_matfile("yeo_RS7_Schafer200S.mat")
    self_Ident_orig = np.zeros((echoes_total_num, subjects_num), dtype=float)
    self_Ident_opt = np.zeros((echoes_total_num, subjects_num), dtype=float)

    ### For each echo-pair, use PCA method to get optimal principle components for matrix reconstruction.
    for echo_test in tqdm(range(echoes_total_num), desc='echo1', leave=False): # Another echo_index: for optimal combination.
        for echo_retest in tqdm(range(echoes_total_num), desc='echo2', leave=False):
            # Imat_orig, Imat_opt, FC_list_test_opt, FC_list_retest_opt, PCA_comps_range, \
            # Idiff_orig, Idiff_recon, Idiff_opt, Iself_orig, Iself_opt, Iothers_orig, Iothers_opt \
            #     = pca_denoising(echo_test, echo_retest, orig_matrixs_test, orig_matrixs_retest)
            
            FC_list_test = orig_matrixs_test[echo_test].T
            FC_list_retest = orig_matrixs_retest[echo_retest].T
            Imat_orig, Idiff_orig, Iself_orig, Iothers_orig = compute_Imat_from_FCs_twodatasets(FC_list_test, FC_list_retest)
            list_Idiff = compute_Idiff_PCA_two_datasets_from_FCs(FC_list_test, FC_list_retest)
            Idiff_opt = np.max(list_Idiff[:,1])
            m_star = int(list_Idiff[np.where(list_Idiff[:,1] == Idiff_opt)[0][0], 0])
            Imat_opt, Idiff_opt, Iself_opt, Iothers_opt = compute_Imat_PCA_opt_two_datasets_from_FCs(FC_list_test, FC_list_retest, list_Idiff)
            
            FC_list_test_opt, FC_list_retest_opt = compute_FCs_PCA_opt_two_datasets_from_FCs(FC_list_test,FC_list_retest,m_star)

            Idiff_recon = list_Idiff[:,1]
            PCA_comps_range = list_Idiff[:,0].astype(np.int32)

            Idiff_mat_orig[echo_retest, echo_test] = Idiff_orig 
            Iself_mat_orig[echo_retest, echo_test] = Iself_orig 
            Iothers_mat_orig[echo_retest, echo_test] = Iothers_orig 
            Idiff_mat_opt[echo_retest, echo_test] = Idiff_opt 
            Iself_mat_opt[echo_retest, echo_test] = Iself_opt 
            Iothers_mat_opt[echo_retest, echo_test] = Iothers_opt  

            draw_echo_pair_results(echo_pair_result_path, Imat_orig, Imat_opt, PCA_comps_range, 
                    Idiff_orig, Idiff_recon, Idiff_opt, echo_test, echo_retest, echo_optcomb) 

            if echo_test == echo_retest:
                mask_diag = np.diag(np.full(subjects_num, True, dtype=bool))
                self_orig = Imat_orig[mask_diag]
                self_Ident_orig[echo_test] = self_orig
                self_opt = Imat_opt[mask_diag]
                self_Ident_opt[echo_test] = self_opt

            # if if_icc or if_save_FC:
                # FC_test_recon, FC_retest_recon = FC_reconstruct(recon_FC_root_img_path, echo_test, echo_retest, echoes_total_num, \
                #         FC_list_test_opt, FC_list_retest_opt, FCs_test, FCs_retest, if_save_FC)
                # if echo_test == echo_retest:
                #     FCs_test_recon[echo_test] = FC_test_recon
                #     FCs_retest_recon[echo_retest] = FC_retest_recon
            if if_icc:
                FCs = np.stack((FC_list_test,FC_list_retest), axis=1)
                ICC_matrix = compute_ICC_idenfiability_from_single_FCs(FCs)
                FCs_opt = np.stack((FC_list_test_opt,FC_list_retest_opt), axis=1)
                ICC_matrix_opt = compute_ICC_idenfiability_from_single_FCs(FCs_opt)
                if echo_test == echo_retest:
                    save_ICC_mat_and_violins(os.path.join(ICC_path, 'ICC_echo'+str(echo_test+1)+'.jpg'), ICC_matrix,yeoOrder,limit_yeo)
                    save_ICC_mat_and_violins(os.path.join(ICC_path, 'ICC_echo'+str(echo_test+1)+'_opt.jpg'), ICC_matrix_opt,yeoOrder,limit_yeo)
            
            
     # Save self identifiabilities before denoising and after denoising. 
    self_Ident_violin_path = os.path.join(result_path, "echo_pair_violin")
    if not os.path.exists(self_Ident_violin_path):
        os.mkdir(self_Ident_violin_path)

    draw_self_Ident_violin(self_Ident_violin_path, self_Ident_orig, self_Ident_opt)           
        
    # Save Idiff Iself Iothers echo-pair matrix before denoising and after denoising. 
    Idiff_root_path = os.path.join(result_path, "Idiff_root")
    if not os.path.exists(Idiff_root_path):
        os.mkdir(Idiff_root_path)

    Idiff_image_path = os.path.join(Idiff_root_path, "Idiff_Iself_Iothers_images")
    if not os.path.exists(Idiff_image_path):
        os.mkdir(Idiff_image_path)

    draw_Idiff_Iself_Iothers(Idiff_image_path, Idiff_mat_orig, Idiff_mat_opt, Iself_mat_orig, Iself_mat_opt, Iothers_mat_orig, Iothers_mat_opt)
    save_Idiff_Iself_Iothers_txt(Idiff_root_path, Idiff_mat_orig, Idiff_mat_opt, Iself_mat_orig, Iself_mat_opt, Iothers_mat_orig, Iothers_mat_opt)

    # if if_icc:
    #     # Compute edgewise ICC with original and optimal reconstruction.
    #     ICC_path = os.path.join(result_path, "ICC")
    #     if not os.path.exists(ICC_path):
    #         os.mkdir(ICC_path)
    #     ICCs = calculate_ICC(FCs_test, FCs_retest)
    #     ICCs_recon = calculate_ICC(FCs_test_recon, FCs_retest_recon)
    #     for echo_index in range(echoes_total_num):
    #         draw_ICC(ICC_path, ICCs, ICCs_recon, echo_index, echo_optcomb)

if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    result_path = args.result_path
    subjects_num = args.subjects_num
    bootstrap_num = args.bootstrap_num
    if_icc = args.icc
    if_save_FC = args.save_FC
    if bootstrap_num == 1:
        fingerprinting_speedup(data_path, result_path, subjects_num, if_icc, if_save_FC)
    else:
        for i in tqdm(range(bootstrap_num), desc='bootstraping count', leave=False):
            fingerprinting_speedup(data_path, result_path, subjects_num, if_icc, if_save_FC)