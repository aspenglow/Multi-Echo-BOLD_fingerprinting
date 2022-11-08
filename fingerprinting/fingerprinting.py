import numpy as np
import os
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr
from tqdm import tqdm

### set global variables.
data_path = "./data"
result_path = "./fingerprinting/results"
echoes_total_num = 4
subjects_total_num = int(len(os.listdir(data_path)) / (echoes_total_num + 1)) # there is another optimal TS.
print("There are " + str(subjects_total_num) + " subjects with " + str(echoes_total_num) + " echoes.")

if not os.path.exists(result_path):
    os.mkdir(result_path)

### load data, compute FC and build map of orig_matrix.

# echoes_FCs: 4-dimensional array.
#   dims:  0: echo_index.  1: subject index.  2,3: FC matrixs
# echoes_orig_matrixs: 3-dimensional array. 
#   dims:  0: echo_index.  1: flatted FC vectors. 2: subjects index
'''
    For original matrix in each echo, each column is a flatted tril FC vector, 
and different columns mean different subjects. Same as the data in MATLAB.
'''
FC_side_length = -1
print("Loading time series from data path...")
data_lists = os.listdir(data_path)
data_lists.sort()
for file in tqdm(data_lists, desc='data', leave=False):
    if "echo-" not in file: 
        continue
    echo_index = file[file.find("echo-")+5 : file.find("echo-")+6]
    echo_index = int(echo_index) - 1
    # Load time series data
    data = np.genfromtxt(fname=os.path.join(data_path, file), dtype='float32', delimiter=' ')
    # Calculate functional connectivity (FC) of a time series
    FC = np.corrcoef(data)
    FC[np.isnan(FC)] = 0.
    # Flatten low triangle of a FC matrix to a vector
    mask = np.tril(np.full((FC.shape[0], FC.shape[0]), True, dtype=bool), -1)  
    orig_column = FC[mask]
    if FC_side_length < 0:
        # Initialization
        FC_side_length = FC.shape[0]
        FC_shape = (FC_side_length, FC_side_length)
        echoes_count = np.zeros(echoes_total_num, dtype=int)
        echoes_FCs_shape = (echoes_total_num, subjects_total_num, FC_side_length, FC_side_length)
        echoes_FCs = np.zeros(echoes_FCs_shape)
        echoes_orig_matrixs_shape = (echoes_total_num, orig_column.size, subjects_total_num)
        echoes_orig_matrixs = np.zeros(echoes_orig_matrixs_shape)

    subject_index = echoes_count[echo_index]
    echoes_FCs[echo_index, subject_index] = FC
    echoes_orig_matrixs[echo_index, :, subject_index] = orig_column 
    echoes_count[echo_index] += 1

# Don't need to manually demean the original matrix because when calculating PCA, the function will first demean it automatically.
echoes_orig_matrixs_mean = np.mean(echoes_orig_matrixs, axis=1) # Calculate mean of each subject and echo

# Initialize reconstructed FC array.
# echoes_FCs_recon: 4-dimensional array.
#   dims:  0: echo_index.  1: subject index.  2,3: reconstructed FC matrixs
echoes_FCs_recon = np.zeros(echoes_FCs_shape)

### For each echo-pair, use PCA method to get optimal principle components for matrix reconstruction.
for echo_index1 in range(echoes_total_num):
    for echo_index2 in range(echoes_total_num):
        if echo_index1 > echo_index2:
            continue
        str_echo_index1 = str(echo_index1+1)
        str_echo_index2 = str(echo_index2+1)
        print("Calculating result with " + str_echo_index1 + "-" + str_echo_index2 + " echo pair.")

        orig_matrix1 = echoes_orig_matrixs[echo_index1]
        orig_matrix2 = echoes_orig_matrixs[echo_index2]
        orig_matrix = np.zeros((orig_matrix1.shape[0], 2*orig_matrix1.shape[1])) 
        orig_matrix[:,0::2] = orig_matrix1
        orig_matrix[:,1::2] = orig_matrix2 # Cross merge each column in orig_matrix1 and orig_matrix2
        max_numPCs = 2 * subjects_total_num
        mask_diag = np.diag(np.full(subjects_total_num, True, dtype=bool))

        # Compute Identifiability matrix, original FCs
        Ident_mat_orig = np.zeros((subjects_total_num, subjects_total_num))
        for i in range(subjects_total_num):
            for j in range(subjects_total_num):
                Ident_mat_orig[i,j] = pearsonr(orig_matrix2[:,i], orig_matrix1[:,j]).statistic

        # Idiff computation, original Identifiability matrix
        Iself_orig = np.mean(Ident_mat_orig[mask_diag])
        Iothers_orig = np.mean(Ident_mat_orig[~mask_diag])
        Idiff_orig = (Iself_orig - Iothers_orig) * 100

        # Differential Identifiability (Idiff) evaluation of PCA decomposition into FC-modes
        # Use PCA method to reconstruct original matrix with diffrent number of principle components.
        '''
            Notice that the pca.fit_transform get pca based on columns, which is the same as in MATLAB
        '''
        Idiff_recon = np.zeros(max_numPCs-1)
        PCA_comps_range = np.array(range(2,max_numPCs+1))
        pca = PCA(n_components=max_numPCs)
        pca.fit(orig_matrix)
        FC_modes = pca.components_.transpose()
        projected_FC_modes = pca.transform(orig_matrix)
        for n in tqdm(PCA_comps_range, desc='PC number', leave=False):
            recon_matrix = np.dot(projected_FC_modes[:,0:n], FC_modes[:,0:n].transpose())
            # Add mean to each column of reconstructed matrix.
            for subject_index in range(subjects_total_num):
                recon_matrix[:, 2*subject_index] += echoes_orig_matrixs_mean[echo_index1, subject_index]
                recon_matrix[:, 2*subject_index+1] += echoes_orig_matrixs_mean[echo_index2, subject_index]
            # Split recon_matrix with different echo. 
            recon_matrix1 = recon_matrix[:,0::2]
            recon_matrix2 = recon_matrix[:,1::2]
            # Compute Identifiability matrix, reconstructed FCs with different number of PCs
            Ident_mat_recon = np.zeros((subjects_total_num, subjects_total_num))
            for i in range(subjects_total_num):
                for j in range(subjects_total_num):
                    Ident_mat_recon[i,j] = pearsonr(recon_matrix2[:,i], recon_matrix1[:,j]).statistic
            # Idiff computation, reconstructed FCs
            Iself_recon = np.mean(Ident_mat_recon[mask_diag])
            Iothers_recon = np.mean(Ident_mat_recon[~mask_diag])
            Idiff_recon[n-2] = (Iself_recon - Iothers_recon) * 100

        # Identifiability matrix at optimal reconstruction
        Idiff_opt = np.max(Idiff_recon)
        m_star = PCA_comps_range[Idiff_recon == Idiff_opt][0]
        pca = PCA(n_components=m_star)
        recon_matrix_opt = pca.inverse_transform(pca.fit_transform(orig_matrix))
        recon_matrix_opt1 = recon_matrix_opt[:,0::2]
        recon_matrix_opt2 = recon_matrix_opt[:,1::2]
        # Compute Recon Identifiability matrix at optimal point
        Ident_mat_recon_opt = np.zeros((subjects_total_num, subjects_total_num))
        for i in range(subjects_total_num):
            for j in range(subjects_total_num):
                Ident_mat_recon_opt[i,j] = pearsonr(recon_matrix_opt2[:,i], recon_matrix_opt1[:,j]).statistic

        # Reconstruct FC from recon_matrix
        # Each FC will have overlaps with # of echoes. We consider to average those overlaps. 
        for subject_index in range(subjects_total_num):
            FC_recon1 = np.identity(FC_side_length)
            FC_recon1[mask] = recon_matrix_opt1[:,subject_index]
            FC_recon1[mask.transpose()] = recon_matrix_opt1[:,subject_index] 
            echoes_FCs_recon[echo_index1][subject_index] += FC_recon1
            
            FC_recon2 = np.identity(FC_side_length)
            FC_recon2[mask] = recon_matrix_opt2[:,subject_index]
            FC_recon2[mask.transpose()] = recon_matrix_opt2[:,subject_index]
            echoes_FCs_recon[echo_index2][subject_index] += FC_recon2 # Each FC will have overlaps with # of echoes. Need to divide by it in the end.

        ### Draw related results
        fig, (ax0, ax1, ax2) = plt.subplots(1,3)
        c = ax0.pcolor(Ident_mat_orig)
        ax0.set_title('original Ident matrix')
        ax0.invert_yaxis()
        ax0.set_aspect('equal', adjustable='box')
        ax0.set_xlabel('Subjects (echo ' + str_echo_index2 + ')') 
        ax0.set_ylabel('Subjects (echo ' + str_echo_index1 + ')')
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.spines['top'].set_position(('data', 0))
        fig.colorbar(c, ax=ax0, orientation='vertical')

        ax1.plot(PCA_comps_range, Idiff_orig*np.ones(PCA_comps_range.size), '--r', label='original data')
        ax1.plot(PCA_comps_range, Idiff_recon, '-b', label="reconstruction data")
        ax1.plot(m_star, Idiff_opt, '-sk', label="optimal")
        ax1.set_title('Idiff assessment based on PCA decomposition (optimal #')
        ax1.axis('tight')
        ax1.legend() 
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.set_xlabel('Number of PCA components')
        ax1.set_ylabel('IDiff (%)')

        c = ax2.pcolor(Ident_mat_recon_opt)
        ax2.set_title('Optimal reconstruction')
        ax2.invert_yaxis()
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_xlabel('Subjects (echo ' + str_echo_index2 + ')') 
        ax2.set_ylabel('Subjects (echo ' + str_echo_index1 + ')')
        ax2.set_xticks([])
        ax2.set_yticks([])
        fig.colorbar(c, ax=ax2, orientation='vertical')

        plt.tight_layout()
        # plt.show()
        print("optimal number of PCs: " + str(m_star) + " optimal IDiff: " + str(Idiff_opt) + "%")
        plt.savefig(os.path.join(result_path, "Result_with_echo_" + str_echo_index1 + "&" + str_echo_index2 + ".jpg"))

# Get final result of reconstructed FCs and save those results.
# FC Need to divide by # of echoes because of overlaps.
echoes_FCs_recon /= echoes_total_num
recon_FC_root_path = os.path.join(result_path, "reconstructed FCs")
if not os.path.exists(recon_FC_root_path):
    os.mkdir(recon_FC_root_path)

print("saving reconstructed FCs...")
for echo_index in tqdm(range(echoes_total_num), leave=False):
    for subject_index in tqdm(range(subjects_total_num), leave=False):
        FC_recon = echoes_FCs_recon[echo_index, subject_index]
        recon_FC_path = os.path.join(recon_FC_root_path, "subject-" + str(subject_index) + "_echo" + str(echo_index))
        np.savetxt(recon_FC_path, FC_recon)
print("succeed.")
