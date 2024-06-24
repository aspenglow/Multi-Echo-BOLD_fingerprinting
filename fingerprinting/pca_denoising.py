import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from tqdm import tqdm

def pca_denoising(echo_test, echo_retest, orig_matrixs_test, orig_matrixs_retest):
    orig_matrixs_test_mean = np.mean(orig_matrixs_test, axis=1) # Calculate mean of each subject and echo
    orig_matrixs_retest_mean = np.mean(orig_matrixs_retest, axis=1)
    # print("Calculating result with " + str(echo_test+1) + "-" + str(echo_retest+1) + " echo pair.")
    orig_matrix_test = orig_matrixs_test[echo_test]
    orig_matrix_retest = orig_matrixs_retest[echo_retest]
    subjects_num = orig_matrix_test.shape[1]

    orig_matrix = np.zeros((orig_matrix_test.shape[0], 2*orig_matrix_test.shape[1])) 
    orig_matrix[:,0::2] = orig_matrix_test
    orig_matrix[:,1::2] = orig_matrix_retest # Cross merge each column in orig_matrix_test and orig_matrix_retest
    
    mask_diag = np.diag(np.full(subjects_num, True, dtype=bool))

    # Compute Identifiability matrix, original FCs
    Ident_mat_orig = np.zeros((subjects_num, subjects_num))
    for i in range(subjects_num):
        for j in range(subjects_num):
            Ident_mat_orig[i,j] = pearsonr(orig_matrix_retest[:,i], orig_matrix_test[:,j]).statistic

    # Idiff computation, original Identifiability matrix
    Iself_orig = np.mean(Ident_mat_orig[mask_diag]) * 100
    Iothers_orig = np.mean(Ident_mat_orig[~mask_diag]) * 100
    Idiff_orig = Iself_orig - Iothers_orig

    # Differential Identifiability (Idiff) evaluation of PCA decomposition into FC-modes
    # Use PCA method to reconstruct original matrix with diffrent number of principle components.
    '''
        Notice that the pca.fit_transform get pca based on columns, which is the same as in MATLAB
    '''
    max_numPCs = 2 * subjects_num
    Idiff_recon = np.zeros(max_numPCs-1)
    PCA_comps_range = np.array(range(2,max_numPCs+1))
    pca = PCA(n_components=max_numPCs)
    pca.fit(orig_matrix)
    FC_modes = pca.components_.transpose()
    projected_FC_modes = pca.transform(orig_matrix)
    for n in tqdm(PCA_comps_range, desc='PC number', leave=False):
        recon_matrix = np.dot(projected_FC_modes[:,0:n], FC_modes[:,0:n].transpose())
        # Add mean to each column of reconstructed matrix.
        for subject_index in range(subjects_num):
            recon_matrix[:, 2*subject_index] += orig_matrixs_test_mean[echo_test, subject_index]
            recon_matrix[:, 2*subject_index+1] += orig_matrixs_retest_mean[echo_retest, subject_index]
        # Split recon_matrix with different echo. 
        recon_matrix_test = recon_matrix[:,0::2]
        recon_matrix_retest = recon_matrix[:,1::2]
        # Compute Identifiability matrix, reconstructed FCs with different number of PCs
        Ident_mat_recon = np.zeros((subjects_num, subjects_num))
        for i in range(subjects_num):
            for j in range(subjects_num):
                Ident_mat_recon[i,j] = pearsonr(recon_matrix_retest[:,i], recon_matrix_test[:,j]).statistic
        # Idiff computation, reconstructed FCs
        Iself_recon = np.mean(Ident_mat_recon[mask_diag])
        Iothers_recon = np.mean(Ident_mat_recon[~mask_diag])
        Idiff_recon[n-2] = (Iself_recon - Iothers_recon) * 100

    # Identifiability matrix at optimal reconstruction
    Idiff_opt = np.max(Idiff_recon)   
    
    m_star = PCA_comps_range[Idiff_recon == Idiff_opt][0]
    recon_matrix_opt = np.dot(projected_FC_modes[:,0:m_star], FC_modes[:,0:m_star].transpose())
    recon_matrix_opt_test = recon_matrix_opt[:,0::2]
    recon_matrix_opt_retest = recon_matrix_opt[:,1::2]
    # Compute Recon Identifiability matrix at optimal point
    Ident_mat_recon_opt = np.zeros((subjects_num, subjects_num))
    for i in range(subjects_num):
        for j in range(subjects_num):
            Ident_mat_recon_opt[i,j] = pearsonr(recon_matrix_opt_retest[:,i], recon_matrix_opt_test[:,j]).statistic
    Iself_opt = np.mean(Ident_mat_recon_opt[mask_diag]) * 100
    Iothers_opt = np.mean(Ident_mat_recon_opt[~mask_diag]) * 100
    # print("optimal number of PCs: " + str(m_star) + " optimal IDiff: " + str(Idiff_opt) + "%")
    
    return Ident_mat_orig, Ident_mat_recon_opt, recon_matrix_opt_test, recon_matrix_opt_retest, PCA_comps_range, \
        Idiff_orig, Idiff_recon, Idiff_opt, Iself_orig, Iself_opt, Iothers_orig, Iothers_opt

    