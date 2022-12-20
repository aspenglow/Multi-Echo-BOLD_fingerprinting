import numpy as np
import scipy.stats as stats
from tqdm import tqdm
import time
import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import colors
from matplotlib.patches import Rectangle
import seaborn as sns

from fingerprinting.utils_identifiability import compute_ICC_yeo_net_dataframe
from fingerprinting.draw_results import draw_FC_reconstruction

def compute_FCs(TCs):
    FCs_list=[];
    for subject in range(TCs.shape[0]):
        FC = np.corrcoef(TCs[subject])
        FCs_list.append(FC_matrix_to_list(FC))
    FCs_list = np.array(FCs_list)
    return FCs_list

def FC_matrix_to_list(FC):
    mask = np.tril(np.full((FC.shape[0], FC.shape[1]), True, dtype=bool), -1)  
    FC_list = FC[mask]
    return FC_list

def FC_list_to_matrix(FC_list):
    mask = np.tril(np.full((FC_side_length, FC_side_length), True, dtype=bool), -1) 
    FC_side_length = len(FC_list)
    ICCs_mat = np.identity(FC_side_length, dtype=float)
    ICCs_mat[mask] = FC_list
    ICCs_mat = ICCs_mat.transpose()
    ICCs_mat[mask] = FC_list
    return ICCs_mat 

def FCs_normalize(FCs):
    t_FCs = (FCs - np.mean(FCs, axis=0)) / np.std(FCs, axis=0)
    return t_FCs

def compute_CNR_from_single_TC(TC1, TC2):
    '''
    Compute the CNR of TC1 and TC2

    input: TC1 -> numpy 1D array
           TC2 -> numpy 1D array

    output: CNR of the two TCs
    '''
    std = np.std(np.concatenate((TC1, TC2)))
    abs = np.abs(np.mean(TC1) - np.mean(TC2))
    return abs / std

def compute_CNRs(TCs1, TCs2):
    '''
    Compute the CNR in two matrices

    input: TCs1 -> numpy 2D array
           TCs2 -> numpy 2D array

    output: matrix encoding all the CNR
    '''
    CNR = np.zeros((len(TCs1), len(TCs2)))
    for i in range(len(TCs1)):
        for j in range(len(TCs2)):
            CNR[i,j] = compute_CNR_from_single_TC(TCs1[i], TCs2[j])
    return CNR



def FC_reconstruct(recon_FC_root_img_path, echo_test, echo_retest, echoes_total_num, recon_matrix_opt_test, recon_matrix_opt_retest, FCs_test, FCs_retest, if_save_FC):
    subjects_num = recon_matrix_opt_test.shape[1]
    FC_side_length = FCs_test[0,0].shape[0]
    mask = np.tril(np.full((FC_side_length, FC_side_length), True, dtype=bool), -1) 
    FCs_test_recon = np.zeros(FCs_test.shape[1:])
    FCs_retest_recon = np.zeros(FCs_retest.shape[1:])
    # Reconstruct FC from recon_matrix
    # Each FC will have overlaps with # of echoes. We consider to average those overlaps. 
    for subject_index in tqdm(range(subjects_num), desc='subject', leave=False):
        FC_test_recon = FC_matrix_reconstruct_with_FC_list(recon_matrix_opt_test[:,subject_index])
        FC_retest_recon = FC_matrix_reconstruct_with_FC_list(recon_matrix_opt_retest[:,subject_index])
        
        FCs_test_recon[subject_index] = FC_test_recon
        FCs_retest_recon[subject_index] = FC_retest_recon
        
        if if_save_FC:
            # Save reconstructed FC by each echo pair.
            FC_test_orig = FCs_test[echo_test, subject_index]
            FC_retest_orig = FCs_retest[echo_retest, subject_index]
            draw_FC_reconstruction(recon_FC_root_img_path, FC_test_orig, FC_test_recon, FC_retest_orig, FC_retest_recon, subject_index, echo_test, echo_retest, echo_optcomb=echoes_total_num-1)
        
        return FCs_test_recon, FCs_retest_recon

def compute_tSNR_from_TCs(TCs):
    ''' 
    Function for computing tSNR for each echo with splited time series (TCs).

    input: TCs_test -> first half of time series, a 4D array of size (n_echoes x n_subjects x n_brain_region x half_len_time_series)
           TCs_retest -> second half of time series, a 4D array of size (n_echoes x n_subjects x n_brain_region x half_len_time_series)
           

    output: a 2D array of size (n_echoes x (n_subjects*n_brain_region)), reporting tSNR distribution of each echo.
    '''

    SNRs = np.mean(TCs, axis=3) / np.std(TCs, axis=3)
    SNRs = SNRs.reshape(SNRs.shape[0], SNRs.shape[1] * SNRs.shape[2])
    return SNRs


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

def plot_FC_mat(ICCs_mat,yeo_net=True,yeoOrder=None,limit_yeo=None,
                subcortical=None,percentile=95,ax=None, title=None):
    if ax == None:
        fig=plt.figure(dpi=150)
        ax=plt.subplot(111)
    
    list_colors_yeo_center=['#d9d9d9','#7fc97f', '#beaed4',
                     '#fdc086', '#ffeda0', '#386cb0',
                     '#f0027f','#bf5b17','#d9d910']
    list_colors_yeo=['#969696','#7fc97f', '#beaed4',
                     '#fdc086','#ffeda0', '#386cb0',
                     '#f0027f','#bf5b17']
    cmap = colors.ListedColormap(list_colors_yeo)

    if yeo_net==None:
        ax.matshow(ICCs_mat,cmap=cmap)
    else:
        FC_mat_reodered=ICCs_mat[yeoOrder,:][:,yeoOrder]
        yeo_ICC_mat=np.where(FC_mat_reodered>np.percentile(ICCs_mat,q=percentile),0,np.nan)
        k=1
        for net,limits in limit_yeo.items():
            y0,y1=limits[0],limits[1]
#             print(y0,y1,k)
            yeo_ICC_mat[y0:y1,y0:y1]=np.where(yeo_ICC_mat[y0:y1,y0:y1]==0,k,np.nan)
            k+=1

        ax.matshow(yeo_ICC_mat,cmap=cmap)
        ax.set_xlim(-0,len(ICCs_mat));
        ax.set_ylim(len(ICCs_mat),0);
        s=0
        net_names=['VIS','SM','DA','VA','L','FP','DMN','SC']
        for net,limits in limit_yeo.items():
            y0,y1=limits[0],limits[1]
#             if y0 <= len(ICCs_mat) and y1<= len(ICCs_mat):
                #print(y0,y1)
            rect_upper = Rectangle((len(ICCs_mat)+2,y0),5, y1-y0, color =list_colors_yeo_center[s+1], clip_on=False,zorder=10)
            rect_bottom = Rectangle((y0,len(ICCs_mat)+2),y1-y0,5,color =list_colors_yeo_center[s+1], clip_on=False,zorder=10)
            ax.text(x=1.025*len(ICCs_mat),y=y0+(0.65*(y1-y0)),s=net_names[s],clip_on=False)
            s+=1
            ax.add_patch(rect_upper)
            ax.add_patch(rect_bottom)
            plt.hlines(y0,y0,y1,colors='k',lw=1)
            plt.hlines(y1,y0,y1,colors='k',lw=1)
            plt.vlines(y0,y0,y1,colors='k',lw=1)
            plt.vlines(y1,y0,y1,colors='k',lw=1)
        if title is not None:
            plt.title(title)

def compute_FC_yeo_net_dataframe(ICCs_mat,yeoOrder,limit_yeo):
    FC_mat_reodered=ICCs_mat[yeoOrder,:][:,yeoOrder]
    list_average_FC=[]
    list_yeo_repeated=[]
    s=1
    for net,limits in limit_yeo.items():
        y0,y1=limits[0],limits[1]
        gap=y1-y0
#         if y0 <= len(ICCs_mat) and y1<= len(ICCs_mat):
#         print(y0,y1,len(FC_mat_reodered[y0:y1,y0:y1][~np.eye(gap,dtype=bool)].flatten()),(gap)*(gap-1))
        list_average_FC.extend(FC_mat_reodered[y0:y1,y0:y1][~np.eye(gap,dtype=bool)].flatten())
        list_yeo_repeated.extend(np.repeat(s,(gap)*(gap-1)))
        s+=1
    current_df=pd.DataFrame([list_average_FC,list_yeo_repeated]).T
    current_df=current_df.rename(columns={0:'FC',1:'YeoNet'})
    return(current_df)

def plot_FCs_violins(ICCs_mat,yeoOrder,limit_yeo,ax=None, title=None):
    if ax == None:
        fig=plt.figure(dpi=150)
        ax=plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for echo in range(len(ICCs_mat)):
        current_df = compute_FC_yeo_net_dataframe(ICCs_mat[echo],yeoOrder,limit_yeo)
        if echo == 4:
            current_df["echo"] = "TE-oc"
        else:
            current_df["echo"] = "TE-"+str(echo+1)
        if 'current_dfs' not in dir():
            current_dfs = current_df
        else:
            current_dfs = pd.concat([current_dfs, current_df], ignore_index=True)
    
    sns.violinplot(x=current_dfs['YeoNet'],y=current_dfs['FC'],orient='v',cut=0.5, data=current_dfs, hue="echo",
                alpha=0.5,width=0.7,linewidth=1)

    net_names=list(limit_yeo.keys())
    # Set the x labels
    ax.set_ylabel('FC Values',fontsize=14)
    ax.set_xlabel('')
    ax.set_ylim(-0.5,1)
    ax.set_xticklabels(net_names);
    if title is not None:
        ax.set_title(title)


def plot_ICCs_violins(ICCs_mat,yeoOrder,limit_yeo,ax=None, title=None):
    if ax == None:
        fig=plt.figure(dpi=150)
        ax=plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    for echo in range(len(ICCs_mat)):
        current_df = compute_ICC_yeo_net_dataframe(ICCs_mat[echo],yeoOrder,limit_yeo)
        if echo == 4:
            current_df["echo"] = "TE-oc"
        else:
            current_df["echo"] = "TE-"+str(echo+1)
        if 'current_dfs' not in dir():
            current_dfs = current_df
        else:
            current_dfs = pd.concat([current_dfs, current_df], ignore_index=True)
    
    sns.violinplot(x=current_dfs['YeoNet'],y=current_dfs['ICC'],orient='v',cut=0.5, data=current_dfs, hue="echo",
                alpha=0.5,width=0.7,linewidth=1)

    net_names=list(limit_yeo.keys())
    # Set the x labels
    ax.set_ylabel('ICC Values',fontsize=14)
    ax.set_xlabel('')
    ax.set_ylim(0,1)
    ax.set_xticklabels(net_names);
    if title is not None:
        ax.set_title(title)

# def plot_FC_mat_and_violins(ICCs_mat,yeoOrder,limit_yeo,percentile=95,subcortical=None):
#     fig=plt.figure(figsize=(10,6),dpi=150)
#     gs = GridSpec(6, 2)
#     ax=fig.add_subplot(gs[0:5, 0])
#     plot_FC_mat(ICCs_mat,yeo_net=True,yeoOrder=yeoOrder,subcortical=subcortical,
#                 limit_yeo=limit_yeo,percentile=percentile,ax=ax)

#     ax=fig.add_subplot(gs[1:4, 1])
#     plot_FC_violins(ICCs_mat,yeoOrder=yeoOrder,limit_yeo=limit_yeo,ax=ax)
#     plt.subplots_adjust(wspace=0.32)