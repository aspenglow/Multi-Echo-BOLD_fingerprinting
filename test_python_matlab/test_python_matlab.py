import numpy as np
import os
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr
from tqdm import tqdm
import scipy.io as scio

data_path = 'test_python_matlab/test_data/'
FCs = scio.loadmat(os.path.join(data_path, 'FC_test_retest.mat')).get('FC')

FC_side_length = -1
numFCs = FCs.shape[2]
for FC_index in range(numFCs):
    FC = FCs[:,:,FC_index]
    mask = np.tril(np.full((FCs.shape[0], FC.shape[0]), True, dtype=bool), -1)  
    orig_column = FC[mask]
    orig_matrix_shape = ()
    if FC_side_length < 0:
        # Initialization
        FC_side_length = FC.shape[0]
        orig_matrixs_shape = (orig_column.size, numFCs)
        orig_matrix = np.zeros(orig_matrixs_shape)
    orig_matrix[:,FC_index] = orig_column

# Test orig_matrix
orig_matrix_matlab = scio.loadmat(os.path.join(data_path, 'orig_matrix.mat')).get("orig_matrix")
assert(np.mean(np.abs(orig_matrix - orig_matrix_matlab)) < 1e-6)

# Calculate Identifiability matrix of the orig_matrix
orig_matrix1 = orig_matrix[:,0::2]
orig_matrix2 = orig_matrix[:,1::2]
subjects_total_num = orig_matrix1.shape[1]
mask_diag = np.diag(np.full(subjects_total_num, True, dtype=bool))
Ident_mat_orig = np.zeros((subjects_total_num, subjects_total_num))
for i in range(subjects_total_num):
    for j in range(subjects_total_num):
        Ident_mat_orig[i,j] = pearsonr(orig_matrix2[:,i], orig_matrix1[:,j]).statistic
# Test Identifiability matrix of original
ident_mat_orig_matlab = scio.loadmat(os.path.join(data_path, 'Ident_mat_orig.mat')).get("Ident_mat_orig")
assert(np.max(np.abs(Ident_mat_orig - ident_mat_orig_matlab)) < 1e-6)

# Test Idiff for original
Iself_orig = np.mean(Ident_mat_orig[mask_diag])
Iothers_orig = np.mean(Ident_mat_orig[~mask_diag])
Idiff_orig = (Iself_orig - Iothers_orig) * 100
Idiff_orig_matlab = 16.65984292
assert(np.abs(Idiff_orig - Idiff_orig_matlab) < 1e-6)

# Calculate PCA and reconstruction
max_numPCs = numFCs
Idiff_recon = np.zeros(max_numPCs-1)
PCA_comps_range = np.array(range(2,max_numPCs+1))
pca = PCA(n_components=max_numPCs)
pca.fit(orig_matrix)
FC_modes = pca.components_.transpose()
projected_FC_modes = pca.transform(orig_matrix)
for n in PCA_comps_range:
    recon_matrix = np.dot(projected_FC_modes[:,0:n], FC_modes[:,0:n].transpose())
    # Add mean to each column of reconstructed matrix.
    for i in range(numFCs):
        recon_matrix[:, i] += np.mean(orig_matrix, axis=0)[i]
    # Test recon_matrix when # of PCs == 20
    if n == 20:
        recon_matrix_PC_20_matlab = scio.loadmat(os.path.join(data_path, 'recon_matrix_PC_20.mat')).get("recon_matrix")
        assert(np.mean(np.abs(recon_matrix - recon_matrix_PC_20_matlab)) < 1e-6)

    # Split recon_matrix with different echo. 
    recon_matrix1 = recon_matrix[:,0::2]
    recon_matrix2 = recon_matrix[:,1::2]
    # Compute Identifiability matrix, reconstructed FCs with different number of PCs
    Ident_mat_recon = np.zeros((subjects_total_num, subjects_total_num))
    for i in range(subjects_total_num):
        for j in range(subjects_total_num):
            Ident_mat_recon[i,j] = pearsonr(recon_matrix2[:,i], recon_matrix1[:,j]).statistic
    # Test recon_matrix when # of PCs == 20
    if n == 20:
        Ident_mat_recon_PC_20_matlab = scio.loadmat(os.path.join(data_path, 'Ident_mat_recon_PC_20.mat')).get("Ident_mat_recon")
        assert(np.mean(np.abs(Ident_mat_recon - Ident_mat_recon_PC_20_matlab)) < 1e-6)

    # Idiff computation, reconstructed FCs
    Iself_recon = np.mean(Ident_mat_recon[mask_diag])
    Iothers_recon = np.mean(Ident_mat_recon[~mask_diag])
    Idiff_recon[n-2] = (Iself_recon - Iothers_recon) * 100
    # Test Idiff_recon when # of PCs == 20
    if n == 20:
        Idiff_recon_PC_20_matlab = 27.7580042
        assert(np.abs(Idiff_recon[n-2] - Idiff_recon_PC_20_matlab) < 1e-6)

# Identifiability matrix at optimal reconstruction
Idiff_opt = np.max(Idiff_recon)
m_star = PCA_comps_range[Idiff_recon == Idiff_opt][0]
# Test Idiff_opt and m_star
Idiff_opt_matlab = 28.4448694
m_star_matlab = 22
assert(np.abs(Idiff_opt - Idiff_opt_matlab) < 1e-6)
assert(np.abs(m_star - m_star_matlab) < 1)

# Calculate optimal reconstruction
pca = PCA(n_components=m_star)
recon_matrix_opt = np.dot(projected_FC_modes[:,0:m_star], FC_modes[:,0:m_star].transpose())
# Test recon_matrix_opt
recon_matrix_opt_matlab = scio.loadmat(os.path.join(data_path, 'recon_matrix_opt.mat')).get("recon_matrix_opt")
assert(np.mean(np.abs(recon_matrix_opt - recon_matrix_opt_matlab)) < 1e-6)

recon_matrix_opt1 = recon_matrix_opt[:,0::2]
recon_matrix_opt2 = recon_matrix_opt[:,1::2]
# Compute Recon Identifiability matrix at optimal point
Ident_mat_recon_opt = np.zeros((subjects_total_num, subjects_total_num))
for i in range(subjects_total_num):
    for j in range(subjects_total_num):
        Ident_mat_recon_opt[i,j] = pearsonr(recon_matrix_opt2[:,i], recon_matrix_opt1[:,j]).statistic
# Test Ident_mat_recon_opt
Ident_mat_recon_opt_matlab = scio.loadmat(os.path.join(data_path, 'Ident_mat_recon_opt.mat')).get("Ident_mat_recon_opt")
assert(np.mean(np.abs(Ident_mat_recon_opt - Ident_mat_recon_opt_matlab)) < 1e-6)

print("succeed.")