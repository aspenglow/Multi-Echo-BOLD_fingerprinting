from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import colors
import seaborn as sns
import os
import numpy as np


def draw_echo_pair_results(echo_pair_result_path, Ident_mat_orig, Ident_mat_recon_opt, PCA_comps_range, Idiff_orig, Idiff_recon, Idiff_opt, echo_test, echo_retest, echo_optcomb):
    str_echo_index_test = str(echo_test + 1)
    str_echo_index_retest = str(echo_retest + 1)
    m_star = PCA_comps_range[Idiff_recon == Idiff_opt][0]
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
    is_test_optcomb = echo_test == echo_optcomb
    is_retest_optcomb = echo_retest == echo_optcomb
    if is_test_optcomb:
        ax0.set_xlabel('Subjects (opt comb test)', fontdict=font) 
    else:
        ax0.set_xlabel('Subjects (echo ' + str_echo_index_test + ' test)', fontdict=font) 
    if is_retest_optcomb:
        ax0.set_ylabel('Subjects (opt comb retest)', fontdict=font)
    else:
        ax0.set_ylabel('Subjects (echo ' + str_echo_index_retest + ' retest)', fontdict=font)
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
    if is_test_optcomb:
        ax1.set_xlabel('Subjects (opt comb test)', fontdict=font) 
    else:
        ax1.set_xlabel('Subjects (echo ' + str_echo_index_test + ' test)', fontdict=font)
    if is_retest_optcomb:
        ax1.set_ylabel('Subjects (opt comb retest)', fontdict=font)
    else:
        ax1.set_ylabel('Subjects (echo ' + str_echo_index_retest + ' retest)', fontdict=font)
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
    mask_diag = np.diag(np.full(Ident_mat_orig.shape[0], True, dtype=bool))
    sns.violinplot([Ident_mat_orig[mask_diag], Ident_mat_recon_opt[mask_diag]])
    ax3.set_title('Self Identifiability', fontdict=font)
    ax3.set_xticks([0,1])
    ax3.set_xticklabels(["Orig", "Recon"])
    ax3.tick_params(labelsize=15)
    ax3.set_ylim(0, 1.3)

    if is_test_optcomb:
        str_echo_index_test = "optcomb"
    if is_retest_optcomb:
        str_echo_index_retest = "optcomb"
    plt.savefig(os.path.join(echo_pair_result_path, "echo_" + str_echo_index_test + "&" + str_echo_index_retest + ".jpg"))
    plt.close()


def draw_FC_reconstruction(recon_FC_root_img_path, FC_test_orig, FC_test_recon, FC_retest_orig, FC_retest_recon, subject_index, echo_test, echo_retest, echo_optcomb):
    str_echo_index_test = str(echo_test + 1)
    str_echo_index_retest = str(echo_retest + 1)
    str_subject_index = str(subject_index + 1)

    vmin = min(np.min(FC_test_orig), np.min(FC_test_recon))
    vmax = max(np.max(FC_test_orig), np.max(FC_test_recon))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(1,2, figsize=(18,10))
    fig.dpi = 100
    font = {'size':20}
    ax0 = ax[0]
    c = ax0.pcolor(FC_test_orig, norm=norm)
    ax0.set_title('Original FC (test)', fontdict=font)
    ax0.invert_yaxis()
    ax0.set_aspect('equal', adjustable='box')
    ax0.set_xlabel('Brain regions', fontdict=font) 
    ax0.set_ylabel('Brain regions', fontdict=font)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.spines['top'].set_position(('data', 0))
    
    ax1 = ax[1]
    c = ax1.pcolor(FC_test_recon, norm=norm)
    ax1.set_title('Recontructed FC (test)', fontdict=font)
    ax1.invert_yaxis()
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel('Brain regions', fontdict=font) 
    ax1.set_ylabel('Brain regions', fontdict=font)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_position(('data', 0))

    cb = fig.colorbar(c, ax=ax, orientation='vertical')
    cb.ax.tick_params(labelsize=15)

    recon_FC_front_img_path = os.path.join(recon_FC_root_img_path, "subject-" + str_subject_index + 
            "_echo" + str_echo_index_test + "_test_with_pairs_" + str_echo_index_test+"front"+str_echo_index_retest+"back.jpg")
    plt.savefig(recon_FC_front_img_path)
    plt.close()

    vmin = min(np.min(FC_retest_orig), np.min(FC_retest_recon))
    vmax = max(np.max(FC_retest_orig), np.max(FC_retest_recon))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(1,2, figsize=(18,10))
    fig.dpi = 100
    font = {'size':20}
    ax0 = ax[0]
    c = ax0.pcolor(FC_test_orig, norm=norm)
    ax0.set_title('Original FC (retest)', fontdict=font)
    ax0.invert_yaxis()
    ax0.set_aspect('equal', adjustable='box')
    ax0.set_xlabel('Brain regions', fontdict=font) 
    ax0.set_ylabel('Brain regions', fontdict=font)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.spines['top'].set_position(('data', 0))
    
    ax1 = ax[1]
    c = ax1.pcolor(FC_test_recon, norm=norm)
    ax1.set_title('Recontructed FC (retest)', fontdict=font)
    ax1.invert_yaxis()
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel('Brain regions', fontdict=font) 
    ax1.set_ylabel('Brain regions', fontdict=font)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_position(('data', 0))

    is_test_optcomb = echo_test == echo_optcomb
    is_retest_optcomb = echo_retest == echo_optcomb

    if is_test_optcomb:
        str_echo_index_test = "optcomb"
    if is_retest_optcomb:
        str_echo_index_retest = "optcomb" 

    recon_FC_back_img_path = os.path.join(recon_FC_root_img_path, "subject-" + str_subject_index + 
            "_echo-" + str_echo_index_retest + "_retest_with_pair_" + str_echo_index_test+"&"+str_echo_index_retest+".jpg")
    plt.savefig(recon_FC_back_img_path)
    plt.close()

def draw_Idiff_Iself_Iothers(image_path, Idiff_mat_orig, Idiff_mat_opt, Iself_mat_orig, Iself_mat_opt, Iothers_mat_orig, Iothers_mat_opt):
    Idiff_min = min(np.min(Idiff_mat_opt), np.min(Idiff_mat_orig))
    Idiff_max = max(np.max(Idiff_mat_opt), np.max(Idiff_mat_orig))
    norm_Idiff = colors.Normalize(vmin=Idiff_min, vmax=Idiff_max)

    fig = plt.figure(figsize=(18,10), dpi=100)
    font = {'size':20}

    fig.dpi = 100
    left, bottom, width, height = 0.0, 0.55, 0.3, 0.3
    ax0 = fig.add_axes([left, bottom, width, height])
    c0 = ax0.pcolor(Idiff_mat_orig, norm=norm_Idiff)
    ax0.set_title('Idiff before PCA denoising', fontdict=font)
    ax0.invert_yaxis()
    ax0.set_aspect('equal', adjustable='box')
    ax0.set_xlabel('Echoes (test)', fontdict=font) 
    ax0.set_ylabel('Echoes (retest)', fontdict=font)
    ax0.set_xticks([])
    ax0.set_yticks([])

    left, bottom, width, height = 0.0, 0.15, 0.3, 0.3
    ax1 = fig.add_axes([left, bottom, width, height])
    c2 = ax1.pcolor(Idiff_mat_opt, norm=norm_Idiff)
    ax1.set_title('Idiff after PCA denoising', fontdict=font)
    ax1.invert_yaxis()
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel('Echoes (test)', fontdict=font) 
    ax1.set_ylabel('Echoes (retest)', fontdict=font)
    ax1.set_xticks([])
    ax1.set_yticks([])

    cb = fig.colorbar(c0, ax=[ax0,ax1], orientation='vertical')
    cb.ax.tick_params(labelsize=15)

    Iself_min = min(np.min(Iself_mat_opt), np.min(Iself_mat_orig))
    Iself_max = max(np.max(Iself_mat_opt), np.max(Iself_mat_orig))
    norm_Iself = colors.Normalize(vmin=Iself_min, vmax=Iself_max)

    left, bottom, width, height = 0.32, 0.55, 0.3, 0.3
    ax2 = fig.add_axes([left, bottom, width, height])
    c0 = ax2.pcolor(Iself_mat_orig, norm=norm_Iself)
    ax2.set_title('Iself before PCA denoising', fontdict=font)
    ax2.invert_yaxis()
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlabel('Echoes (test)', fontdict=font) 
    ax2.set_ylabel('Echoes (retest)', fontdict=font)
    ax2.set_xticks([])
    ax2.set_yticks([])

    left, bottom, width, height = 0.32, 0.15, 0.3, 0.3
    ax3 = fig.add_axes([left, bottom, width, height])
    c2 = ax3.pcolor(Iself_mat_opt, norm=norm_Iself)
    ax3.set_title('Iself after PCA denoising', fontdict=font)
    ax3.invert_yaxis()
    ax3.set_aspect('equal', adjustable='box')
    ax3.set_xlabel('Echoes (test)', fontdict=font) 
    ax3.set_ylabel('Echoes (retest)', fontdict=font)
    ax3.set_xticks([])
    ax3.set_yticks([])

    cb = fig.colorbar(c0, ax=[ax2,ax3], orientation='vertical')
    cb.ax.tick_params(labelsize=15)

    Iothers_min = min(np.min(Iothers_mat_opt), np.min(Iothers_mat_orig))
    Iothers_max = max(np.max(Iothers_mat_opt), np.max(Iothers_mat_orig))
    norm_Iothers = colors.Normalize(vmin=Iothers_min, vmax=Iothers_max)

    left, bottom, width, height = 0.64, 0.55, 0.3, 0.3
    ax4 = fig.add_axes([left, bottom, width, height])
    c0 = ax4.pcolor(Iothers_mat_orig, norm=norm_Iothers)
    ax4.set_title('Iothers before PCA denoising', fontdict=font)
    ax4.invert_yaxis()
    ax4.set_aspect('equal', adjustable='box')
    ax4.set_xlabel('Echoes (test)', fontdict=font) 
    ax4.set_ylabel('Echoes (retest)', fontdict=font)
    ax4.set_xticks([])
    ax4.set_yticks([])
    
    left, bottom, width, height = 0.64, 0.15, 0.3, 0.3
    ax5 = fig.add_axes([left, bottom, width, height])
    c2 = ax5.pcolor(Iothers_mat_opt, norm=norm_Iothers)
    ax5.set_title('Iothers after PCA denoising', fontdict=font)
    ax5.invert_yaxis()
    ax5.set_aspect('equal', adjustable='box')
    ax5.set_xlabel('Echoes (test)', fontdict=font) 
    ax5.set_ylabel('Echoes (retest)', fontdict=font)
    ax5.set_xticks([])
    ax5.set_yticks([])

    cb = fig.colorbar(c0, ax=[ax4,ax5], orientation='vertical')
    cb.ax.tick_params(labelsize=15)

    plt.savefig(os.path.join(image_path, "Idiff_Iself_Iothers.jpg"))
    plt.close()

def draw_echo_pairs_self_violin(image_path, self_elements_orig, self_elements_opt):
    fig = plt.figure(figsize=(18,10), dpi=100)
    font = {'size':20}
    left, bottom, width, height = 0.05, 0.55, 0.9, 0.4
    ax0 = fig.add_axes([left, bottom, width, height])
    sns.violinplot([self_elements_orig[0], self_elements_orig[1], self_elements_orig[2], self_elements_orig[3], self_elements_orig[4], \
                    self_elements_orig[5], self_elements_orig[6], self_elements_orig[7], self_elements_orig[8], self_elements_orig[9]])
    ax0.set_title('Self Identifiabilities before PCA denoising', fontdict=font)
    ax0.set_xticks([0,1,2,3,4,5,6,7,8,9])
    ax0.set_xticklabels(["2-1", "3-1", "4-1", "oc-1", "3-2", "4-2", "oc-2", "4-3", "oc-3", "oc-4"])
    ax0.tick_params(labelsize=15)
    ax0.set_ylim(0.2, 1.2)

    left, bottom, width, height = 0.05, 0.05, 0.9, 0.4
    ax1 = fig.add_axes([left, bottom, width, height])
    sns.violinplot([self_elements_opt[0], self_elements_opt[1], self_elements_opt[2], self_elements_opt[3], self_elements_opt[4], \
                    self_elements_opt[5], self_elements_opt[6], self_elements_opt[7], self_elements_opt[8], self_elements_opt[9]])
    ax1.set_title('Self Identifiabilities after PCA denoising', fontdict=font)
    ax1.set_xticks([0,1,2,3,4,5,6,7,8,9])
    ax1.set_xticklabels(["2-1", "3-1", "4-1", "oc-1", "3-2", "4-2", "oc-2", "4-3", "oc-3", "oc-4"])
    ax1.tick_params(labelsize=15)
    ax1.set_ylim(0.2, 1.2)

    plt.savefig(os.path.join(image_path, "Self_identifiability_echopairs.jpg"))
    plt.close()

def draw_ICC(ICC_path, ICCs, ICCs_recon, echo_index, echo_optcomb):
    ICCs_min = min(np.min(ICCs[echo_index]), np.min(ICCs_recon[echo_index]))
    ICCs_max = max(np.max(ICCs[echo_index]), np.max(ICCs_recon[echo_index]))
    
    norm = colors.Normalize(vmin=ICCs_min, vmax=ICCs_max)
    fig = plt.figure(figsize=(18,10), dpi=100)
    font = {'size':20}

    fig.dpi = 100
    left, bottom, width, height = 0.05, 0.2, 0.4, 0.4
    ax0 = fig.add_axes([left, bottom, width, height])
    c0 = ax0.pcolor(ICCs[echo_index], norm=norm)
    ax0.set_title('ICC - echo ' + str(echo_index+1) + ' test Orig', fontdict=font)
    ax0.invert_yaxis()
    ax0.set_aspect('equal', adjustable='box')
    ax0.set_xlabel('Brain regions', fontdict=font) 
    ax0.set_ylabel('Brain regions', fontdict=font)
    ax0.set_xticks([])
    ax0.set_yticks([])

    left, bottom, width, height = 0.55, 0.2, 0.4, 0.4
    ax1 = fig.add_axes([left, bottom, width, height])
    c1 = ax1.pcolor(ICCs_recon[echo_index], norm=norm)
    ax1.set_title('ICC - echo ' + str(echo_index+1) + ' test Recon', fontdict=font)
    ax1.invert_yaxis()
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel('Brain regions', fontdict=font) 
    ax1.set_ylabel('Brain regions', fontdict=font)
    ax1.set_xticks([])
    ax1.set_yticks([])

    cb = fig.colorbar(c0, ax=[ax0,ax1], orientation='vertical')
    cb.ax.tick_params(labelsize=15)
    if echo_index == echo_optcomb:
        plt.savefig(os.path.join(ICC_path, "ICC_echo_optcomb.jpg"))
    else:
        plt.savefig(os.path.join(ICC_path, "ICC_echo_" + str(echo_index+1) + ".jpg"))
    plt.close()
