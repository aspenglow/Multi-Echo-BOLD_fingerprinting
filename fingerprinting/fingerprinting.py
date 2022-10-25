import numpy as np
import os
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

### set global variables.
data_path = "./data/"
result_path = "./fingerprinting/results/"
echoes_total_num = 4
subjects_total_num = 4

if not os.path.exists(data_path):
    os.mkdir(data_path)

if not os.path.exists(result_path):
    os.mkdir(result_path)

### load data, compute FC and build map of orig_matrix.

# echoes_FCs: 3-dimensional array.
#   dims:  0: echo_index.  1,2 : FC matrixs
# echoes_orig_matrixs: 3-dimensional array. 
#   dims:  0: echo_index.  1: subject index.  2: flatten FC vectors.

FC_side_length = -1
for file in os.listdir(data_path):
    if "echo-" not in file: 
        continue
    echo_index = file[file.find("echo-")+5 : file.find("echo-")+6]
    echo_index = int(echo_index) - 1
    # Load time series data
    data = np.genfromtxt(fname=os.path.join(data_path, file), dtype='float32', delimiter=' ')
    # Calculate functional connectivity (FC) of a time series
    FC = np.corrcoef(data)
    # Flatten low triangle of a FC matrix to a vector
    mask = np.tril(np.full((FC.shape[0], FC.shape[0]), True, dtype=bool), -1)  
    orig_row = FC[mask]
    if FC_side_length < 0:
        # Initialization
        FC_side_length = FC.shape[0]
        echoes_count = np.zeros(echoes_total_num, dtype=int)
        echoes_FCs_shape = (echoes_total_num, FC_side_length, FC_side_length)
        echoes_FCs = np.zeros(echoes_FCs_shape)
        echoes_orig_matrixs_shape = (echoes_total_num, subjects_total_num, orig_row.size)
        echoes_orig_matrixs = np.zeros(echoes_orig_matrixs_shape)

    echoes_FCs[echo_index] = FC
    subject_index = echoes_count[echo_index]
    echoes_orig_matrixs[echo_index][subject_index] = orig_row 
    echoes_count[echo_index] += 1
    
### For each echo-pair, use PCA method to get optimal principle components for matrix reconstruction.
for echo_index1 in range(echoes_total_num):
    for echo_index2 in range(echoes_total_num):
        if echo_index1 >= echo_index2:
            continue
        orig_matrix1 = echoes_orig_matrixs[echo_index1]
        orig_matrix2 = echoes_orig_matrixs[echo_index2]
        orig_matrix = np.concatenate((orig_matrix1, orig_matrix2))
        max_numPCs = 2 * subjects_total_num
        mask_diag = np.diag(np.full(subjects_total_num, True, dtype=bool))
        PCA_comps_range = np.array(range(2,max_numPCs+1))

        # Compute Identifiability matrix, original FCs
        Ident_mat_orig = np.zeros((subjects_total_num, subjects_total_num))
        for i in range(subjects_total_num):
            for j in range(subjects_total_num):
                Ident_mat_orig[i,j] = pearsonr(orig_matrix1[i,:], orig_matrix2[j,:]).statistic

        # Idiff computation, original FCs
        Iself_orig = np.mean(Ident_mat_orig[mask_diag])
        Iothers_orig = np.mean(Ident_mat_orig[~mask_diag])
        Idiff_orig = (Iself_orig - Iothers_orig) * 100

        # Differential Identifiability (Idiff) evaluation of PCA decomposition into FC-modes
        Idiff_recon = np.zeros(max_numPCs)
        for n in PCA_comps_range:
            pca = PCA(n_components=n)
            recon_matrix = pca.inverse_transform(pca.fit_transform(orig_matrix))
            recon_matrix1 = recon_matrix[0:subjects_total_num,:]
            recon_matrix2 = recon_matrix[subjects_total_num:,:]
            # Compute Identifiability matrix, reconstructed FCs with different number of PCs
            # Ident_mat_recon = np.corr(recon_matrix1, recon_matrix2, rowvar=False)
            Ident_mat_recon = np.zeros((subjects_total_num, subjects_total_num))
            for i in range(subjects_total_num):
                for j in range(subjects_total_num):
                    Ident_mat_recon[i,j] = pearsonr(recon_matrix1[i,:], recon_matrix2[j,:]).statistic
            # Idiff computation, reconstructed FCs
            Iself_recon = np.mean(Ident_mat_recon[mask_diag])
            Iothers_recon = np.mean(Ident_mat_recon[~mask_diag])
            Idiff_recon[n-1] = (Iself_recon - Iothers_recon) * 100

        # Identifiability matrix at optimal reconstruction
        Idiff_opt = np.max(Idiff_recon)
        m_star = PCA_comps_range[Idiff_recon[1:] == Idiff_opt][0]
        pca = PCA(n_components=m_star)
        recon_matrix_opt = pca.inverse_transform(pca.fit_transform(orig_matrix))
        recon_matrix_opt1 = recon_matrix_opt[0:subjects_total_num,:]
        recon_matrix_opt2 = recon_matrix_opt[subjects_total_num:,:]
        # Compute Recon Identifiability matrix at optimal point
        Ident_mat_recon_opt = np.zeros((subjects_total_num, subjects_total_num))
        for i in range(subjects_total_num):
            for j in range(subjects_total_num):
                Ident_mat_recon_opt[i,j] = pearsonr(recon_matrix_opt1[i,:], recon_matrix_opt2[j,:]).statistic

        ### Draw related results
        fig, (ax0, ax1, ax2) = plt.subplots(1,3)
        c = ax0.pcolor(Ident_mat_orig)
        ax0.set_title('original Ident matrix')
        ax0.invert_yaxis()
        ax0.set_aspect('equal', adjustable='box')
        ax0.set_xlabel('Subject')
        ax0.set_ylabel('Subject')
        ax0.axis('off')

        c = ax1.plot(PCA_comps_range, Idiff_recon[1:])
        ax1.plot(PCA_comps_range, Idiff_orig*np.ones(PCA_comps_range.size), )
        ax1.plot(m_star, Idiff_opt, '-sk')
        ax1.set_title('Idiff assessment based on PCA decomposition')
        ax1.axis('tight')

        ax1.set_xlabel('Number of PCA components')
        ax1.set_ylabel('IDiff (%)')

        c = ax2.pcolor(Ident_mat_recon_opt)
        ax2.set_title('Optimal reconstruction')
        ax2.invert_yaxis()
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_xlabel('Subject')
        ax2.set_ylabel('Subject')
        ax2.axis('off')

        plt.tight_layout()
        # plt.show()
        plt.savefig(result_path + "Result_with_echo_" + str(echo_index1+1) + "&" + str(echo_index2+1) + ".jpg")
