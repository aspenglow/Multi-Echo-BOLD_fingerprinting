import numpy as np
import scipy.stats as stats
from tqdm import tqdm
import time
import os

from draw_results import draw_FC_reconstruction

def ICC(matrix, alpha=0.05, r0=0):
    '''Intraclass correlation
    matrix is matrix of observations. Each row is an object of measurement and
    each column is a judge or measurement.
    '1-1' is implement: The degree of absolute agreement among measurements made on
    randomly selected objects. It estimates the correlation of any two
    measurements.
    ICC is the estimated intraclass correlation. LB and UB are upper
    and lower bounds of the ICC with alpha level of significance. 

    In addition to estimation of ICC, a hypothesis test is performed
    with the null hypothesis that ICC = r0. The F value, degrees of
    freedom and the corresponding p-value of the this test are reported.

    Translated in python from the matlab code of Arash Salarian, 2008

    Reference: McGraw, K. O., Wong, S. P., "Forming Inferences About
    Some Intraclass Correlation Coefficients", Psychological Methods,
    Vol. 1, No. 1, pp. 30-46, 1996'''

    M = np.array(matrix)
    n, k = np.shape(M)
    SStotal = np.var(M.flatten(), ddof=1) * (n * k - 1)
    MSR = np.var(np.mean(M, 1), ddof=1) * k
    MSW = np.sum(np.var(M, 1, ddof=1)) / n
    MSC = np.var(np.mean(M, 0), ddof=1) * n
    MSE = (SStotal - MSR * (n - 1) - MSC * (k - 1)) / ((n - 1) * (k - 1))
    # print(n, k)
    # print("SStotal", SStotal)
    # print("MSR", MSR)
    # print("MSW", MSW)
    # print("MSC", MSC)
    # print("MSE", MSE)

    r = (MSR - MSW) / (MSR + (k - 1) * MSW)

    F = (MSR / MSW) * (1 - r0) / (1 + (k - 1) * r0)
    df1 = n - 1
    df2 = n * (k - 1)
    p = 1 - stats.f.cdf(F, df1, df2)

    FL = (MSR / MSW) * (stats.f.isf(1 - alpha / 2, n * (k - 1), n - 1))
    FU = (MSR / MSW) / (stats.f.isf(1 - alpha / 2, n - 1, n * (k - 1)))

    LB = (FL - 1) / (FL + (k - 1))
    UB = (FU - 1) / (FU + (k - 1))
    return (r, LB, UB, F, df1, df2, p)

def FC_reconstruct(recon_FC_root_img_path, echo_test, echo_retest, echoes_total_num, recon_matrix_opt_test, recon_matrix_opt_retest, FCs_test, FCs_retest):
    subjects_num = recon_matrix_opt_test.shape[1]
    FC_side_length = FCs_test[0,0].shape[0]
    mask = np.tril(np.full((FC_side_length, FC_side_length), True, dtype=bool), -1) 
    # Reconstruct FC from recon_matrix
    # Each FC will have overlaps with # of echoes. We consider to average those overlaps. 
    for subject_index in tqdm(range(subjects_num), desc='subject', leave=False):
        FC_test_recon = np.identity(FC_side_length)
        FC_test_recon[mask] = recon_matrix_opt_test[:,subject_index]
        FC_test_recon = FC_test_recon.transpose()
        FC_test_recon[mask] = recon_matrix_opt_test[:,subject_index] 
        
        FC_retest_recon = np.identity(FC_side_length)
        FC_retest_recon[mask] = recon_matrix_opt_retest[:,subject_index]
        FC_retest_recon = FC_retest_recon.transpose()
        FC_retest_recon[mask] = recon_matrix_opt_retest[:,subject_index]
        
        # Save reconstructed FC by each echo pair.
        FC_test_orig = FCs_test[echo_test, subject_index]
        FC_retest_orig = FCs_retest[echo_retest, subject_index]
        
        draw_FC_reconstruction(recon_FC_root_img_path, FC_test_orig, FC_test_recon, FC_retest_orig, FC_retest_recon, subject_index, echo_test, echo_retest, echo_optcomb=echoes_total_num-1)

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
    