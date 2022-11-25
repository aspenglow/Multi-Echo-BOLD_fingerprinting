import numpy as np
import os
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import colors
import seaborn as sns
from scipy.stats import pearsonr
from tqdm import tqdm
import argparse
from utils import ICC

def parse_args():
    parser = argparse.ArgumentParser(description='PCA denoising with spliting timeseries to test and retest.')
    parser.add_argument('--data_path', default="./data",
                        help='path of the timeseries.')
    parser.add_argument('--result_path', default="./fingerprinting/results",
                        help='sum the integers (default: find the max)')
    args = parser.parse_args()
    return args

### Set global variables.
data_path = "./data"
result_path = "./fingerprinting/results"
echoes_total_num = 4
subjects_total_num = int(len(os.listdir(data_path)) / (echoes_total_num + 1)) # there is another optimal TS.
print("There are " + str(subjects_total_num) + " subjects with " + str(echoes_total_num) + " echoes.")

if not os.path.exists(result_path):
    os.mkdir(result_path)

### Load data, compute FC and build map of orig_matrix.

# echoes_FCs: 4-dimensional array.
#   dims:  0: echo_index.  1: subject index.  2,3: FC matrixs
# echoes_orig_matrixs_front, echoes_orig_matrixs_back: 3-dimensional array. For first half and second half of TS, respectively.
#   dims:  0: echo_index.  1: flatted FC vectors. 2: subject index
'''
    For original matrix in each echo, each column is a flatted tril FC vector, 
and different columns mean different subjects. Same as the data in MATLAB.
'''
FC_side_length = -1
print("Loading time series from data path...")
data_lists = os.listdir(data_path)
data_lists.sort()
for file in tqdm(data_lists, desc='data', leave=False):
    is_opt_comb = "echo-" not in file
    subject_id = file[file.find("sub-")+4 : file.find("_task-")]
    if not is_opt_comb: 
        echo_index = file[file.find("echo-")+5 : file.find("echo-")+6]
        echo_index = int(echo_index) - 1
    # Load time series data
    data = np.genfromtxt(fname=os.path.join(data_path, file), dtype='float32', delimiter=' ')
    TS_length = data.shape[1]
    TS_front = data[:, :int(TS_length/2)]
    TS_back = data[:, int(TS_length/2):]
    # Calculate functional connectivity (FC) of a time series
    FC_front = np.corrcoef(TS_front)
    FC_back = np.corrcoef(TS_back)
    assert(np.sum(np.isnan(FC_front)) == 0) # make sure that all the FCs are valid.
    assert(np.sum(np.isnan(FC_back)) == 0)
    assert(np.sum(np.isinf(FC_front)) == 0)
    assert(np.sum(np.isinf(FC_back)) == 0)
    # Flatten low triangle of a FC matrix to a vector
    mask = np.tril(np.full((FC_front.shape[0], FC_front.shape[0]), True, dtype=bool), -1)  
    orig_column_front = FC_front[mask]
    orig_column_back = FC_back[mask]
    if FC_side_length < 0:
        # Initialization
        FC_side_length = FC_front.shape[0]
        FC_shape = (FC_side_length, FC_side_length)
        echoes_count = np.zeros(echoes_total_num, dtype=int)
        echoes_FCs_shape = (echoes_total_num, subjects_total_num, FC_side_length, FC_side_length)
        echoes_FCs_front = np.zeros(echoes_FCs_shape)
        echoes_FCs_back = np.zeros(echoes_FCs_shape)
        echoes_ICCs_shape = (echoes_total_num, FC_side_length, FC_side_length)
        echoes_ICCs_front = np.zeros(echoes_ICCs_shape)
        echoes_ICCs_back = np.zeros(echoes_ICCs_shape)
        echoes_ICCs_front_recon = np.zeros(echoes_ICCs_shape)
        echoes_ICCs_back_recon = np.zeros(echoes_ICCs_shape)

        optcomb_FCs_shape = echoes_FCs_shape[1:]
        optcomb_FCs_front = np.zeros(optcomb_FCs_shape)
        optcomb_FCs_back = np.zeros(optcomb_FCs_shape)
        optcomb_ICCs_shape = echoes_ICCs_shape[1:]
        optcomb_ICCs = np.zeros(optcomb_ICCs_shape)
        optcomb_ICCs_recon = np.zeros(optcomb_ICCs_shape)
        
        echoes_orig_matrixs_shape = (echoes_total_num, orig_column_front.size, subjects_total_num)
        echoes_orig_matrixs_front = np.zeros(echoes_orig_matrixs_shape)
        echoes_orig_matrixs_back = np.zeros(echoes_orig_matrixs_shape)
        
        optcomb_orig_matrixs_shape = echoes_orig_matrixs_shape[1:]
        optcomb_orig_matrixs_front = np.zeros(optcomb_orig_matrixs_shape)
        optcomb_orig_matrixs_back = np.zeros(optcomb_orig_matrixs_shape)

    
    if is_opt_comb: # For optimal combination
        subject_index = echoes_count[0] - 1
        optcomb_FCs_front[subject_index] = FC_front
        optcomb_orig_matrixs_front[:, subject_index] = orig_column_front
        optcomb_FCs_back[subject_index] = FC_back
        optcomb_orig_matrixs_back[:,subject_index] = orig_column_back
    else:
        subject_index = echoes_count[echo_index]
        echoes_FCs_front[echo_index, subject_index] = FC_front
        echoes_orig_matrixs_front[echo_index, :, subject_index] = orig_column_front 
        echoes_FCs_back[echo_index, subject_index] = FC_back
        echoes_orig_matrixs_back[echo_index, :, subject_index] = orig_column_back
        echoes_count[echo_index] += 1

# Don't need to manually demean the original matrix because when calculating PCA, the function will first demean it automatically.
echoes_orig_matrixs_front_mean = np.mean(echoes_orig_matrixs_front, axis=1) # Calculate mean of each subject and echo
echoes_orig_matrixs_back_mean = np.mean(echoes_orig_matrixs_back, axis=1)
optcomb_orig_matrixs_front_mean = np.mean(optcomb_orig_matrixs_front, axis=0) 
optcomb_orig_matrixs_back_mean = np.mean(optcomb_orig_matrixs_back, axis=0)

# Initialize reconstructed FC array.
# echoes_FCs_front_recon: 4-dimensional array.
#   dims:  0: echo_index.  1: subject index.  2,3: reconstructed FC matrixs
echoes_FCs_front_recon = np.zeros(echoes_FCs_shape)
echoes_FCs_back_recon = np.zeros(echoes_FCs_shape)
optcomb_FCs_front_recon = np.zeros(optcomb_FCs_shape)
optcomb_FCs_back_recon = np.zeros(optcomb_FCs_shape)
# Initialize optimal Idiff matrix. 
# #dims: 0: echo_front index.  1: echo_back index.  Idiff_opt_matrix[i,j]: optimal Idiff for echo_front i, echo_back j
Idiff_orig_matrix = np.zeros((echoes_total_num, echoes_total_num), dtype=float)
Idiff_opt_matrix = np.zeros((echoes_total_num, echoes_total_num), dtype=float)
opt_comb_Idiff_orig = np.zeros((1, 1), dtype=float)
opt_comb_Idiff_opt = np.zeros((1, 1), dtype=float)

echo_pair_result_path = os.path.join(result_path, "echo_pair_results")
if not os.path.exists(echo_pair_result_path):
    os.mkdir(echo_pair_result_path)
recon_FC_root_img_path = os.path.join(result_path, "reconstructed_FCs_images")
if not os.path.exists(recon_FC_root_img_path):
    os.mkdir(recon_FC_root_img_path)
### For each echo-pair, use PCA method to get optimal principle components for matrix reconstruction.
for echo_index_front in range(echoes_total_num+1): # Another echo_index: for optimal combination.
    for echo_index_back in range(echoes_total_num):
        is_opt_comb = echo_index_front == echoes_total_num 
        is_end = is_opt_comb & echo_index_back > 0 # Have computed optimal combination circumstance.
        if is_end:
            break
        if is_opt_comb: 
            print("Calculating result with optimal combination.")
            orig_matrix_front = optcomb_orig_matrixs_front
            orig_matrix_back = optcomb_orig_matrixs_back
        else:
            str_echo_index_front = str(echo_index_front+1)
            str_echo_index_back = str(echo_index_back+1)
            print("Calculating result with " + str(echo_index_front+1) + "-" + str(echo_index_back+1) + " echo pair.")
            orig_matrix_front = echoes_orig_matrixs_front[echo_index_front]
            orig_matrix_back = echoes_orig_matrixs_back[echo_index_back]
        
        orig_matrix = np.zeros((orig_matrix_front.shape[0], 2*orig_matrix_front.shape[1])) 
        orig_matrix[:,0::2] = orig_matrix_front
        orig_matrix[:,1::2] = orig_matrix_back # Cross merge each column in orig_matrix_front and orig_matrix_back
        max_numPCs = 2 * subjects_total_num
        mask_diag = np.diag(np.full(subjects_total_num, True, dtype=bool))

        # Compute Identifiability matrix, original FCs
        Ident_mat_orig = np.zeros((subjects_total_num, subjects_total_num))
        for i in range(subjects_total_num):
            for j in range(subjects_total_num):
                Ident_mat_orig[i,j] = pearsonr(orig_matrix_back[:,i], orig_matrix_front[:,j]).statistic

        # Idiff computation, original Identifiability matrix
        Iself_orig = np.mean(Ident_mat_orig[mask_diag])
        Iothers_orig = np.mean(Ident_mat_orig[~mask_diag])
        Idiff_orig = (Iself_orig - Iothers_orig) * 100
        if is_opt_comb:
            opt_comb_Idiff_orig[0,0] = Idiff_orig
        else:
            Idiff_orig_matrix[echo_index_back, echo_index_front] = Idiff_orig

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
            if is_opt_comb:
                for subject_index in range(subjects_total_num):
                    recon_matrix[:, 2*subject_index] += optcomb_orig_matrixs_front_mean[subject_index]
                    recon_matrix[:, 2*subject_index+1] += optcomb_orig_matrixs_back_mean[subject_index]
            else:
                for subject_index in range(subjects_total_num):
                    recon_matrix[:, 2*subject_index] += echoes_orig_matrixs_front_mean[echo_index_front, subject_index]
                    recon_matrix[:, 2*subject_index+1] += echoes_orig_matrixs_back_mean[echo_index_back, subject_index]
            # Split recon_matrix with different echo. 
            recon_matrix_front = recon_matrix[:,0::2]
            recon_matrix_back = recon_matrix[:,1::2]
            # Compute Identifiability matrix, reconstructed FCs with different number of PCs
            Ident_mat_recon = np.zeros((subjects_total_num, subjects_total_num))
            for i in range(subjects_total_num):
                for j in range(subjects_total_num):
                    Ident_mat_recon[i,j] = pearsonr(recon_matrix_back[:,i], recon_matrix_front[:,j]).statistic
            # Idiff computation, reconstructed FCs
            Iself_recon = np.mean(Ident_mat_recon[mask_diag])
            Iothers_recon = np.mean(Ident_mat_recon[~mask_diag])
            Idiff_recon[n-2] = (Iself_recon - Iothers_recon) * 100

        # Identifiability matrix at optimal reconstruction
        Idiff_opt = np.max(Idiff_recon)
        if is_opt_comb:
            opt_comb_Idiff_opt[0, 0] = Idiff_opt
        else:
            Idiff_opt_matrix[echo_index_back, echo_index_front] = Idiff_opt # Save Idiff_opt into Idiff_opt_matrix 
        
        m_star = PCA_comps_range[Idiff_recon == Idiff_opt][0]
        recon_matrix_opt = np.dot(projected_FC_modes[:,0:m_star], FC_modes[:,0:m_star].transpose())
        recon_matrix_opt_front = recon_matrix_opt[:,0::2]
        recon_matrix_opt_back = recon_matrix_opt[:,1::2]
        # Compute Recon Identifiability matrix at optimal point
        Ident_mat_recon_opt = np.zeros((subjects_total_num, subjects_total_num))
        for i in range(subjects_total_num):
            for j in range(subjects_total_num):
                Ident_mat_recon_opt[i,j] = pearsonr(recon_matrix_opt_back[:,i], recon_matrix_opt_front[:,j]).statistic

        # Reconstruct FC from recon_matrix
        # Each FC will have overlaps with # of echoes. We consider to average those overlaps. 
        for subject_index in tqdm(range(subjects_total_num), desc='subject', leave=False):
            str_subject_index = str(subject_index+1)
        
            FC_front_recon = np.identity(FC_side_length)
            FC_front_recon[mask] = recon_matrix_opt_front[:,subject_index]
            FC_front_recon = FC_front_recon.transpose()
            FC_front_recon[mask] = recon_matrix_opt_front[:,subject_index] 
            
            FC_back_recon = np.identity(FC_side_length)
            FC_back_recon[mask] = recon_matrix_opt_back[:,subject_index]
            FC_back_recon = FC_back_recon.transpose()
            FC_back_recon[mask] = recon_matrix_opt_back[:,subject_index]

            if is_opt_comb:
                optcomb_FCs_front_recon[subject_index] = FC_front_recon
                optcomb_FCs_back_recon[subject_index] = FC_back_recon
            else:
                # Each FC will have overlaps with # of echoes. Need to divide by it in the end.
                echoes_FCs_front_recon[echo_index_front][subject_index] += FC_front_recon
                echoes_FCs_back_recon[echo_index_back][subject_index] += FC_back_recon 
            
            # Save reconstructed FC by each echo pair.
            if is_opt_comb:
                FC_front_orig = optcomb_FCs_front[subject_index]
                FC_back_orig = optcomb_FCs_back[subject_index]
            else:
                FC_front_orig = echoes_FCs_front[echo_index_front, subject_index]
                FC_back_orig = echoes_FCs_back[echo_index_back, subject_index]
            
            vmin = min(np.min(FC_front_orig), np.min(FC_front_recon))
            vmax = max(np.max(FC_front_orig), np.max(FC_front_recon))
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

            fig, ax = plt.subplots(1,2, figsize=(18,10))
            fig.dpi = 100
            font = {'size':20}
            ax0 = ax[0]
            c = ax0.pcolor(FC_front_orig, norm=norm)
            ax0.set_title('Original FC', fontdict=font)
            ax0.invert_yaxis()
            ax0.set_aspect('equal', adjustable='box')
            ax0.set_xlabel('Brain regions', fontdict=font) 
            ax0.set_ylabel('Brain regions', fontdict=font)
            ax0.set_xticks([])
            ax0.set_yticks([])
            ax0.spines['top'].set_position(('data', 0))
            
            ax1 = ax[1]
            c = ax1.pcolor(FC_front_recon, norm=norm)
            ax1.set_title('Recontructed FC with echo pair ' + str_echo_index_front + "&" + str_echo_index_back, fontdict=font)
            ax1.invert_yaxis()
            ax1.set_aspect('equal', adjustable='box')
            ax1.set_xlabel('Brain regions', fontdict=font) 
            ax1.set_ylabel('Brain regions', fontdict=font)
            ax1.set_xticks([])
            ax1.set_yticks([])
            cb = fig.colorbar(c, ax=[ax0, ax1], orientation='vertical')
            cb.ax.tick_params(labelsize=15)

            if is_opt_comb:
                recon_FC_front_img_path = os.path.join(recon_FC_root_img_path, "subject-" + str_subject_index + 
                        "opt_comb_front.jpg")
            else:
                recon_FC_front_img_path = os.path.join(recon_FC_root_img_path, "subject-" + str_subject_index + 
                        "_echo" + str_echo_index_front + "_front_with_pairs_" + str_echo_index_front+"front"+str_echo_index_back+"back.jpg")
            plt.savefig(recon_FC_front_img_path)

            vmin = min(np.min(FC_back_orig), np.min(FC_back_recon))
            vmax = max(np.max(FC_back_orig), np.max(FC_back_recon))
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            c0 = ax0.pcolor(FC_back_orig, norm=norm)
            c1 = ax1.pcolor(FC_back_recon, norm=norm)

            if is_opt_comb:
                recon_FC_back_img_path = os.path.join(recon_FC_root_img_path, "subject-" + str_subject_index + 
                        "opt_comb_back.jpg")
            else:
                recon_FC_back_img_path = os.path.join(recon_FC_root_img_path, "subject-" + str_subject_index + 
                        "_echo" + str_echo_index_back + "_back_with_pair_" + str_echo_index_front+"front"+str_echo_index_back+"back.jpg")
            plt.savefig(recon_FC_back_img_path)
            plt.close()

        ### Draw related results for each echo-pair
        vmin = min(np.min(Ident_mat_orig), np.min(Ident_mat_recon_opt))
        vmax = max(np.max(Ident_mat_orig), np.max(Ident_mat_recon_opt))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        fig = plt.figure(figsize=(18,10), dpi=100)
        font = {'size':15}
        left, bottom, width, height = -0.01, 0.55, 0.4, 0.4
        ax0 = fig.add_axes([left, bottom, width, height])
        c0 = ax0.pcolor(Ident_mat_orig, norm=norm)
        ax0.set_title('Original Ident Matrix', fontdict=font)
        ax0.invert_yaxis()
        ax0.set_aspect('equal', adjustable='box')
        if is_opt_comb:
            ax0.set_xlabel('Subjects (opt comb test)', fontdict=font) 
            ax0.set_ylabel('Subjects (opt comb retest)', fontdict=font)
        else:
            ax0.set_xlabel('Subjects (echo ' + str_echo_index_front + ' test)', fontdict=font) 
            ax0.set_ylabel('Subjects (echo ' + str_echo_index_back + ' retest)', fontdict=font)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.spines['top'].set_position(('data', 0))

        left, bottom, width, height = -0.01, 0.05, 0.4, 0.4
        ax1 = fig.add_axes([left, bottom, width, height])
        c1 = ax1.pcolor(Ident_mat_recon_opt, norm=norm)
        ax1.set_title('Optimal reconstruction', fontdict=font)
        ax1.invert_yaxis()
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_aspect('equal', adjustable='box')
        if is_opt_comb:
            ax1.set_xlabel('Subjects (opt comb test)', fontdict=font) 
            ax1.set_ylabel('Subjects (opt comb retest)', fontdict=font)
        else:
            ax1.set_xlabel('Subjects (echo ' + str_echo_index_front + ' test)', fontdict=font) 
            ax1.set_ylabel('Subjects (echo ' + str_echo_index_back + ' retest)', fontdict=font)
        ax1.set_xticks([])
        ax1.set_yticks([])
        cb = fig.colorbar(c0, ax=[ax0, ax1], orientation='vertical')
        cb.ax.tick_params(labelsize=15)

        left, bottom, width, height = 0.55, 0.55, 0.4, 0.4
        ax2 = fig.add_axes([left, bottom, width, height])
        # ax2 = plt.subplot2grid((1,2), (0,1), rowspan=2)
        ax2.plot(PCA_comps_range, Idiff_orig*np.ones(PCA_comps_range.size), '--r', linewidth=5, label='original data')
        ax2.plot(PCA_comps_range, Idiff_recon, '-b', linewidth=5, label="reconstruction data")
        point = ax2.scatter(m_star, Idiff_opt, c='k', label="optimal", s=150, marker='s')
        point.set_zorder(10)
        ax2.axes.tick_params(labelsize=15)
        ax2.set_title('Idiff assessment based on PCA decomposition', fontdict=font)
        ax2.legend(prop=font) 
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.set_xlabel('Number of PCA components',fontdict=font)
        ax2.set_ylabel('IDiff (%)',fontdict=font)
        
        fig.text(0.7,0.75,"m_star="+str(m_star),fontdict={'size':20})
        fig.text(0.7,0.7,"Idiff_opt="+str(Idiff_opt)[:6]+"%",fontdict={'size':20})

        left, bottom, width, height = 0.55, 0.05, 0.4, 0.4
        ax3 = fig.add_axes([left, bottom, width, height])
        sns.violinplot([Ident_mat_orig[mask_diag], Ident_mat_recon_opt[mask_diag]])
        ax3.set_title('Self Identifiability', fontdict=font)
        ax3.set_xticks([0,1])
        ax3.set_xticklabels(["Orig", "Recon"])
        ax3.tick_params(labelsize=15)
        ax3.set_ylim(0, 1.3)

        print("optimal number of PCs: " + str(m_star) + " optimal IDiff: " + str(Idiff_opt) + "%")
        if is_opt_comb:
            plt.savefig(os.path.join(echo_pair_result_path, "Result_with_optimal_combination.jpg"))
        else:
            plt.savefig(os.path.join(echo_pair_result_path, "Result_with_echo_" + str_echo_index_front + "&" + str_echo_index_back + ".jpg"))
        plt.close()

# Save Idiff_orig_matrix and Idiff_opt_matrix results. 
Idiff_opt_matrix_root_path = os.path.join(result_path, "optimal_Idiff_opt_matrix")
if not os.path.exists(Idiff_opt_matrix_root_path):
    os.mkdir(Idiff_opt_matrix_root_path)

Idiff_min = min(np.min(Idiff_opt_matrix),opt_comb_Idiff_opt[0,0],np.min(Idiff_orig_matrix),opt_comb_Idiff_orig[0,0])
Idiff_max = max(np.max(Idiff_opt_matrix),opt_comb_Idiff_opt[0,0],np.max(Idiff_orig_matrix),opt_comb_Idiff_orig[0,0])
norm = colors.Normalize(vmin=Idiff_min, vmax=Idiff_max)
fig = plt.figure(figsize=(18,10), dpi=100)
font = {'size':20}

fig.dpi = 100
left, bottom, width, height = 0.4, 0.55, 0.4, 0.4
ax0 = fig.add_axes([left, bottom, width, height])
c0 = ax0.pcolor(Idiff_orig_matrix, norm=norm)
ax0.set_title('Original Idiff Matrix', fontdict=font)
ax0.invert_yaxis()
ax0.set_aspect('equal', adjustable='box')
ax0.set_xlabel('Echoes (test)', fontdict=font) 
ax0.set_ylabel('Echoes (retest)', fontdict=font)
ax0.set_xticks([])
ax0.set_yticks([])

left, bottom, width, height = 0.23, 0.65, 0.15, 0.15
ax1 = fig.add_axes([left, bottom, width, height])
c1 = ax1.pcolor(opt_comb_Idiff_orig, norm=norm)
ax1.set_title('Original Optimal Comb Idiff', fontdict=font)
ax1.invert_yaxis()
ax1.set_aspect('equal', adjustable='box')
ax1.set_xticks([])
ax1.set_yticks([])

left, bottom, width, height = 0.4, 0.05, 0.4, 0.4
ax2 = fig.add_axes([left, bottom, width, height])
c2 = ax2.pcolor(Idiff_opt_matrix, norm=norm)
ax2.set_title('After PCA denoising', fontdict=font)
ax2.invert_yaxis()
ax2.set_aspect('equal', adjustable='box')
ax2.set_xlabel('Echoes (test)', fontdict=font) 
ax2.set_ylabel('Echoes (retest)', fontdict=font)
ax2.set_xticks([])
ax2.set_yticks([])

left, bottom, width, height = 0.23, 0.15, 0.15, 0.15
ax3 = fig.add_axes([left, bottom, width, height])
c1 = ax3.pcolor(opt_comb_Idiff_opt, norm=norm)
ax3.set_title('After PCA denoising', fontdict=font)
ax3.invert_yaxis()
ax3.set_aspect('equal', adjustable='box')
ax3.set_xticks([])
ax3.set_yticks([])

cb = fig.colorbar(c0, ax=[ax0,ax1,ax2,ax3], orientation='vertical')
cb.ax.tick_params(labelsize=15)
plt.savefig(os.path.join(Idiff_opt_matrix_root_path, "optimal_Idiff_opt_matrix.jpg"))
plt.close()
  
print("Idiff original matrix:")
print(Idiff_orig_matrix)
print("Original Original Comb Idiff" + str(opt_comb_Idiff_orig))
print("Idiff optimal matrix:")
print(Idiff_opt_matrix)
print("Original Optimal Comb Idiff" + str(opt_comb_Idiff_opt))

# Get final result of reconstructed FCs and save those results.
# FC Need to divide by # of echoes because of overlaps.
echoes_FCs_front_recon /= echoes_total_num
echoes_FCs_back_recon /= echoes_total_num
    
# # Save average of original FCs and reconstructed FCs of subjects
FC_avg_img_path = os.path.join(result_path, "average_FCs")
if not os.path.exists(FC_avg_img_path):
    os.mkdir(FC_avg_img_path)

echoes_FCs_front_avg = np.mean(echoes_FCs_front,axis=1)
echoes_FCs_back_avg = np.mean(echoes_FCs_back,axis=1)
echoes_FCs_front_recon_avg = np.mean(echoes_FCs_front_recon,axis=1)
echoes_FCs_back_recon_avg = np.mean(echoes_FCs_back_recon,axis=1)
for echo_index in range(echoes_total_num):
    font = {'size':20}
    fig, ax = plt.subplots(1, 2, figsize=(18,10))
    fig.dpi = 100
    ax0 = ax[0]
    FC_front_orig = echoes_FCs_front_avg[echo_index]
    FC_front_recon = echoes_FCs_front_recon_avg[echo_index]
    FC_back_orig = echoes_FCs_back_avg[echo_index]
    FC_back_recon = echoes_FCs_back_recon_avg[echo_index]
    
    # front
    vmin = min(np.min(FC_front_orig), np.min(FC_front_recon))
    vmax = max(np.max(FC_front_orig), np.max(FC_front_recon))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    c = ax0.pcolor(FC_front_orig, norm=norm)
    ax0.set_title('Original FC average', fontdict=font)
    ax0.invert_yaxis()
    ax0.set_aspect('equal', adjustable='box')
    ax0.set_xlabel('Brain regions', fontdict=font) 
    ax0.set_ylabel('Brain regions', fontdict=font)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.spines['top'].set_position(('data', 0))
    
    ax1 = ax[1]
    c = ax1.pcolor(FC_front_recon, norm=norm)
    ax1.set_title('Recontructed FC average', fontdict=font)
    ax1.invert_yaxis()
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel('Brain regions', fontdict=font) 
    ax1.set_ylabel('Brain regions', fontdict=font)
    ax1.set_xticks([])
    ax1.set_yticks([])
    cb = fig.colorbar(c, ax=[ax0, ax1], orientation='vertical')
    cb.ax.tick_params(labelsize=15)

    FC_avg_front_path = os.path.join(FC_avg_img_path, "FC_echo_" + str(echo_index+1) + "_front.jpg")
    plt.savefig(FC_avg_front_path)

    # back
    vmin = min(np.min(FC_back_orig), np.min(FC_back_recon))
    vmax = max(np.max(FC_back_orig), np.max(FC_back_recon))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    c = ax0.pcolor(FC_back_orig, norm=norm)
    c = ax1.pcolor(FC_front_recon, norm=norm)

    FC_avg_back_path = os.path.join(FC_avg_img_path, "FC_echo_" + str(echo_index+1) + "_back.jpg")
    plt.savefig(FC_avg_back_path)
    plt.close()
   
# Compute edgewise ICC with original and optimal reconstruction.
ICC_path = os.path.join(result_path, "ICC")
if not os.path.exists(ICC_path):
    os.mkdir(ICC_path)

for echoes_index in tqdm(range(echoes_total_num+1), leave=False):
    for brain_region_1 in range(FC_side_length):
        for brain_region_2 in range(FC_side_length):
            is_opt_comb = echoes_index == echoes_total_num 
            if is_opt_comb: 
                edgewise_front = optcomb_FCs_front[:, brain_region_1, brain_region_2]
                (r_front, _, _, _, _, _, _) = ICC(edgewise_front[np.newaxis, :])
                optcomb_ICCs[brain_region_1, brain_region_2] = r_front
                edgewise_back = optcomb_FCs_back[:, brain_region_1, brain_region_2]
                (r_back, _, _, _, _, _, _) = ICC(edgewise_back[np.newaxis, :])
                optcomb_ICCs_back[brain_region_1, brain_region_2] = r_back

                edgewise_front = optcomb_FCs_front_recon[:, brain_region_1, brain_region_2]
                (r_front, _, _, _, _, _, _) = ICC(edgewise_front[np.newaxis, :])
                optcomb_ICCs_recon[brain_region_1, brain_region_2] = r_front
                edgewise_back = optcomb_FCs_back[:, brain_region_1, brain_region_2]
                (r_back, _, _, _, _, _, _) = ICC(edgewise_back[np.newaxis, :])
                optcomb_ICCs_back_recon[brain_region_1, brain_region_2] = r_back

            else:
                edgewise_front = echoes_FCs_front[echoes_index, :, brain_region_1, brain_region_2]
                (r_front, _, _, _, _, _, _) = ICC(edgewise_front[np.newaxis, :])
                echoes_ICCs_front[echoes_index, brain_region_1, brain_region_2] = r_front
                edgewise_back = echoes_FCs_back[echoes_index, :, brain_region_1, brain_region_2]
                (r_back, _, _, _, _, _, _) = ICC(edgewise_back[np.newaxis, :])
                echoes_ICCs_back[echoes_index, brain_region_1, brain_region_2] = r_back

                edgewise_front = echoes_FCs_front_recon[echoes_index, :, brain_region_1, brain_region_2]
                (r_front, _, _, _, _, _, _) = ICC(edgewise_front[np.newaxis, :])
                echoes_ICCs_front_recon[echoes_index, brain_region_1, brain_region_2] = r_front
                edgewise_back = echoes_FCs_back_recon[echoes_index, :, brain_region_1, brain_region_2]
                (r_back, _, _, _, _, _, _) = ICC(edgewise_back[np.newaxis, :])
                echoes_ICCs_back_recon[echoes_index, brain_region_1, brain_region_2] = r_back

for echoes_index in range(echoes_total_num+1):
    norm = colors.Normalize(vmin=-0.3, vmax=1)
    is_opt_comb = echoes_index == echoes_total_num 
    if is_opt_comb: 
        fig = plt.figure(figsize=(18,10), dpi=100)
        font = {'size':20}

        fig.dpi = 100
        left, bottom, width, height = 0.0, 0.55, 0.4, 0.4
        ax0 = fig.add_axes([left, bottom, width, height])
        c0 = ax0.pcolor(optcomb_ICCs, norm=norm)
        ax0.set_title('ICC - Opt comb test Orig', fontdict=font)
        ax0.invert_yaxis()
        ax0.set_aspect('equal', adjustable='box')
        ax0.set_xlabel('Brain regions', fontdict=font) 
        ax0.set_ylabel('Brain regions', fontdict=font)
        ax0.set_xticks([])
        ax0.set_yticks([])

        left, bottom, width, height = 0.55, 0.55, 0.4, 0.4
        ax1 = fig.add_axes([left, bottom, width, height])
        c1 = ax1.pcolor(optcomb_ICCs_recon, norm=norm)
        ax1.set_title('ICC - Opt comb test Recon', fontdict=font)
        ax1.invert_yaxis()
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_xlabel('Brain regions', fontdict=font) 
        ax1.set_ylabel('Brain regions', fontdict=font)
        ax1.set_xticks([])
        ax1.set_yticks([])

        left, bottom, width, height = 0.0, 0.05, 0.4, 0.4
        ax2 = fig.add_axes([left, bottom, width, height])
        c2 = ax2.pcolor(optcomb_ICCs_back, norm=norm)
        ax2.set_title('ICC - Opt comb retest Orig', fontdict=font)
        ax2.invert_yaxis()
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_xlabel('Brain regions', fontdict=font) 
        ax2.set_ylabel('Brain regions', fontdict=font)
        ax2.set_xticks([])
        ax2.set_yticks([])

        left, bottom, width, height = 0.55, 0.05, 0.4, 0.4
        ax3 = fig.add_axes([left, bottom, width, height])
        c3 = ax3.pcolor(optcomb_ICCs_back_recon, norm=norm)
        ax3.set_title('ICC - Opt comb retest Recon', fontdict=font)
        ax3.invert_yaxis()
        ax3.set_aspect('equal', adjustable='box')
        ax3.set_xlabel('Brain regions', fontdict=font) 
        ax3.set_ylabel('Brain regions', fontdict=font)
        ax3.set_xticks([])
        ax3.set_yticks([])
        cb = fig.colorbar(c0, ax=[ax0,ax1,ax2,ax3], orientation='vertical')
        cb.ax.tick_params(labelsize=15)
        plt.savefig(os.path.join(ICC_path, "ICC_opt_comb.jpg"))
        plt.close()
    else: 
        ICCs_min = min(np.min(echoes_ICCs_front[echoes_index]), np.min(echoes_ICCs_back[echoes_index]), 
                np.min(echoes_ICCs_front_recon[echoes_index]), np.min(echoes_ICCs_back_recon[echoes_index]))
        ICCs_max = max(np.max(echoes_ICCs_front[echoes_index]), np.max(echoes_ICCs_back[echoes_index]), 
                np.max(echoes_ICCs_front_recon[echoes_index]), np.max(echoes_ICCs_back_recon[echoes_index]))
        
        norm = colors.Normalize(vmin=ICCs_min, vmax=ICCs_max)
        fig = plt.figure(figsize=(18,10), dpi=100)
        font = {'size':20}

        fig.dpi = 100
        left, bottom, width, height = 0.0, 0.55, 0.4, 0.4
        ax0 = fig.add_axes([left, bottom, width, height])
        c0 = ax0.pcolor(echoes_ICCs_front[echoes_index], norm=norm)
        ax0.set_title('ICC - echo ' + str(echoes_index+1) + ' test Orig', fontdict=font)
        ax0.invert_yaxis()
        ax0.set_aspect('equal', adjustable='box')
        ax0.set_xlabel('Brain regions', fontdict=font) 
        ax0.set_ylabel('Brain regions', fontdict=font)
        ax0.set_xticks([])
        ax0.set_yticks([])

        left, bottom, width, height = 0.55, 0.55, 0.4, 0.4
        ax1 = fig.add_axes([left, bottom, width, height])
        c1 = ax1.pcolor(echoes_ICCs_front_recon[echoes_index], norm=norm)
        ax1.set_title('ICC - echo ' + str(echoes_index+1) + ' test Recon', fontdict=font)
        ax1.invert_yaxis()
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_xlabel('Brain regions', fontdict=font) 
        ax1.set_ylabel('Brain regions', fontdict=font)
        ax1.set_xticks([])
        ax1.set_yticks([])

        left, bottom, width, height = 0.0, 0.05, 0.4, 0.4
        ax2 = fig.add_axes([left, bottom, width, height])
        c2 = ax2.pcolor(echoes_ICCs_back[echoes_index], norm=norm)
        ax2.set_title('ICC - echo ' + str(echoes_index+1) + ' retest Orig', fontdict=font)
        ax2.invert_yaxis()
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_xlabel('Brain regions', fontdict=font) 
        ax2.set_ylabel('Brain regions', fontdict=font)
        ax2.set_xticks([])
        ax2.set_yticks([])

        left, bottom, width, height = 0.55, 0.05, 0.4, 0.4
        ax3 = fig.add_axes([left, bottom, width, height])
        c3 = ax3.pcolor(echoes_ICCs_back_recon[echoes_index], norm=norm)
        ax3.set_title('ICC - echo ' + str(echoes_index+1) + ' retest Recon', fontdict=font)
        ax3.invert_yaxis()
        ax3.set_aspect('equal', adjustable='box')
        ax3.set_xlabel('Brain regions', fontdict=font) 
        ax3.set_ylabel('Brain regions', fontdict=font)
        ax3.set_xticks([])
        ax3.set_yticks([])
        cb = fig.colorbar(c0, ax=[ax0,ax1,ax2,ax3], orientation='vertical')
        cb.ax.tick_params(labelsize=15)
        plt.savefig(os.path.join(ICC_path, "ICC_echo " + str(echoes_index+1) + ".jpg"))
        plt.close()

print("succeed.")
