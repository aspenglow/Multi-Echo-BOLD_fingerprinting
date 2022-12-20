"""
This file contains fast implementations for computing the Identifiability matrix of one
or two datasets, along with the PCA decomposition method  proposed by E. Amico and J.Goni
in the paper "The quest for identifiability in human functional connectomes"

Authors: Andrea Santoro
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from  multiprocessing import Pool
import matplotlib.pyplot as plt
import sys
from matplotlib.patches import Rectangle
import seaborn as sns
from matplotlib import colors
from matplotlib.gridspec import GridSpec
from fingerprinting.ICC import ICC
from scipy.io import loadmat



#Taken from https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b611bf873bd5836748647221480071a87/sklearn/utils/extmath.py
def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.
    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    Parameters
    ----------
    u : ndarray
        u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
    v : ndarray
        u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
        The input v should really be called vt to be consistent with scipy's
        output.
    u_based_decision : bool, default=True
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.
    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.
    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v


#https://stackoverflow.com/questions/71844846/is-there-a-faster-way-to-get-correlation-coefficents
def pairwise_correlation(A, B):
    '''
    Compute the Pearson correlation coefficient between two matrices

    input: A -> numpy 2D array
           B -> numpy 2D array

    output: matrix encoding all the Pearson correlation'''

    
    am = A - np.mean(A, axis=0, keepdims=True)
    bm = B - np.mean(B, axis=0, keepdims=True)
    return am.T @ bm /  (np.sqrt(
        np.sum(am**2, axis=0,
               keepdims=True)).T * np.sqrt(
        np.sum(bm**2, axis=0, keepdims=True)))


def compute_Iscores(Imat):
    '''
    Function to compute the Idiff, Iself and Iothers from an identifiability matrix Imat

    input: Identiability matrix (nsubjs x nsubjs)

    output: Differential Identifiability score (%),
            Iself
            Iothers
    '''
    Iself=np.mean(Imat.diagonal())
    Iothers=np.mean(Imat[~np.eye(Imat.shape[0],dtype=bool)])
    Idiff=(Iself-Iothers)*100
    return(Idiff,Iself,Iothers)



def computing_FCs_halved(data,trimming_length=0):
    ''' 
    Function for computing Test and Retest  Functional Connectomes from a 
    dictionary in the form {subjID: fMRI data} by splitting each fMRI
    recording in two halves 

    input: data -> dictionary in the form {subjID: fMRI data}
           trimming_length -> default 0 (no trimming), if different from 0 then trim
                              all the recordings to the same length

    output: a 3D array of size (nsubjects x 2 x nedges) reporting the 
            flattened upper triangular matrix of functional connectomes 
            constructed when splitting each fMRI recording in two halves
    '''

    nsubjects=len(data)
    n_ROIs=np.shape(list(data.values())[0])[0]
    
    u,v=np.triu_indices(n=n_ROIs,k=1,m=n_ROIs)
    FC_list=[];
    
    ##Computing the functional connectomes splitting the TC in half
    for x in data.values():
        x=np.array(x)
        if trimming_length!=0:
            TC=x[:,:trimming_length]    
        else:
            TC=x
        n_timepoints=np.shape(TC)[1]
        half_T=int(n_timepoints/2)
        
        FC_1=np.nan_to_num(pairwise_correlation(TC[:,:half_T].T,TC[:,:half_T].T))
        FC_2=np.nan_to_num(pairwise_correlation(TC[:,half_T:].T,TC[:,half_T:].T))
        FC_list.append([FC_1[u,v],FC_2[u,v]])
    FC_list=np.array(FC_list)
    return FC_list


def computing_FCs(data,trimming_length=0):
    '''
    Function for computing Functional Connectomes/Correlation Matrices from a 
    dictionary in the form {subjID: fMRI data}

    input: data -> dictionary in the form {subjID: fMRI data}
           trimming_length -> default 0 (no trimming), if different from 0 then trim
                              all the recordings to the same length

    output: a 2D array of size (nsubjects x nedges) reporting the 
            flattened upper triangular matrix of functional connectomes 
            constructed from the fMRI recordings of each subject'''

    nsubjects=len(data)
    n_ROIs=np.shape(list(data.values())[0])[0]
    
    u,v=np.triu_indices(n=n_ROIs,k=1,m=n_ROIs)
    FC_list=[];

    ##Computing the functional connectomes for each subject
    for x in data.values():
        x=np.array(x)
        if trimming_length!=0:
            TC=x[:,:trimming_length]    
        else:
            TC=x
        FC=np.nan_to_num(pairwise_correlation(TC.T,TC.T))
        FC_list.append(FC[u,v])
    FC_list=np.array(FC_list)
    return (FC_list)



def compute_Imat_from_single_session_TC(data,trimming_length=0,nodes=None):
    '''
    Function to compute Imat, Idiff, Iself and Iothers from a dictionary in the form
    {subjID: fMRI data}.

    input: data -> dictionary computing_FCs_halvedin the form {subjID: fMRI data}
           trimming_length -> default 0 (no trimming), if different from 0 then trim
                              all the recordings to the same length

    output: Identifiability matrix,
            Differential Identifiability score (%),
            Iself
            Iothers
    '''
    nsubjects=len(data)
    n_ROIs=np.shape(list(data.values())[0])[0]
    u,v=np.triu_indices(n=n_ROIs,k=1,m=n_ROIs)
    FC_list=computing_FCs_halved(data,trimming_length)
    ##Compute the Identifiability matrix
    I_mat=pairwise_correlation(FC_list[:,0].T,FC_list[:,1].T)
    Idiff,Iself,Iothers=compute_Iscores(I_mat)
    return(I_mat,Idiff,Iself,Iothers)


def compute_Imat_from_TC_twodatasets(data1,data2,trimming_length=0):
    '''
    Function to compute Imat, Idiff, Iself and Iothers from two dictionaries data1 and 
    data2 in the form {subjID: fMRI data}, data1 is Test, data2 is Retest 

    input: data1 -> dictionary in the form {subjID: fMRI data}
           data2 -> dictionary in the form {subjID: fMRI data}
           trimming_length -> default 0 (no trimming), if different from 0 then trim
                              all the recordings to the same length

    output: Identifiability matrix,
            Differential Identifiability score (%),
            Iself
            Iothers
    '''
    nsubjects=len(data1)
    n_ROIs=np.shape(list(data1.values())[0])[0]
    u,v=np.triu_indices(n=n_ROIs,k=1,m=n_ROIs)
    FC1_list=computing_FCs(data1,trimming_length)
    FC2_list=computing_FCs(data2,trimming_length)
    ##Analogous command of np.array(list(zip(FC1_list,FC2_list)))) but slightly faster
    FC_list=np.stack((FC1_list,FC2_list), axis=1)
    ##Compute the Identifiability matrix
    I_mat=pairwise_correlation(FC_list[:,0].T,FC_list[:,1].T)
    Idiff,Iself,Iothers=compute_Iscores(I_mat)
    return(I_mat,Idiff,Iself,Iothers)    



def compute_Imat_from_single_session_FCs(FC_list):
    '''
    Function to compute Imat, Idiff, Iself and Iothers from a 3D array of FCs
    of the form (nsubjs x 2 x nedges)

    input: data -> numpy array with a shape: nsubjs x 2 x nedges(upper triangular FC)
    
    output: Identifiability matrix,
            Differential Identifiability score (%),
            Iself
            Iothers
    '''
    nsubjects=len(FC_list)
    FC_list=np.array(FC_list)
    ##Computing the Identifiability matrix
    I_mat=pairwise_correlation(FC_list[:,0].T,FC_list[:,1].T)
    Idiff,Iself,Iothers=compute_Iscores(I_mat)
    return(I_mat,Idiff,Iself,Iothers)


def compute_Imat_from_FCs_twodatasets(FC1_list,FC2_list):
    '''
    Function to compute Imat, Idiff, Iself and Iothers from two 2D arrays of size (nsubjects x nedges)
    reporting the flattened upper triangular matrix of functional connectomes 
    constructed from the fMRI recordings of each subject

    input: FC1_list -> a 2D array of size (nsubjects x nedges) with flattened FCs for TEST
           FC2_list -> a 2D array of size (nsubjects x nedges) with flattened FCs for RETEST

    output: Identifiability matrix,
            Differential Identifiability score (%),
            Iself
            Iothers
    '''
    ##Analogous command of np.array(list(zip(FC1_list,FC2_list)))) but slightly faster
    FC_list=np.stack((FC1_list,FC2_list), axis=1)
    ##Compute the Identifiability matrix
    I_mat=pairwise_correlation(FC_list[:,0].T,FC_list[:,1].T)
    Idiff,Iself,Iothers=compute_Iscores(I_mat)
    return(I_mat,Idiff,Iself,Iothers) 



def compute_Imat_from_listFCs(data_FC):
    '''
    Function to compute Imat, Idiff, Iself and Iothers from a 2D array of FCs
    of the form (2*nsubjs x nedges), alternating Test, Retest as odd and even
    rows 

    input: data -> numpy array with a shape: 2*nsubjs x nedges(upper triangular FC)

    output: Identifiability matrix,
            Differential Identifiability score (%),
            Iself
            Iothers
    '''
    nsubj_TR,nlinks=np.shape(data_FC)
    ##nsubj_TR is double of the number of subjects since it contains test and retes
    nsubj=int(nsubj_TR/2)##These are all the number of subjects
    #Reshape the matrix as a 3D array
    data_FC=data_FC.reshape((nsubj,2,nlinks))
    #Compute the Identifiability matrix (test, retest)
    Imat=pairwise_correlation(data_FC[:,0].T,data_FC[:,1].T)
    Idiff,Iself,Iothers=compute_Iscores(Imat)
    return(Imat,Idiff,Iself,Iothers)


def fit_kcomponents(U,S,Vt,all_connectomes,mean_,k_components):
    '''
    Function that reconstruct the matrix "all_connectomes"
    starting from the first k_components of the pca 
    (and decomposition U,S,Vt) passed as input 

    input: U,S,Vt-> Decomposition obtained from the command svd decomposition of all connectomes
           all_connectomes -> 2d numpy array containing all the flattened FC (upper triangular)
                             that was passed as input for the pca._fit(all_connectomes)
           mean_ -> mean of all the connectomes across rows (i.e. np.mean(all_connectomes,axis=0))
           k_components -> integer from 1 to Ncomponents max 

    output: 2D array containing the reconstructed numpy array of "all_connectomes" using only
           k_components
    ''' 
    
    UU = U[:, : k_components]
    UU =UU* S[: k_components]
    reconstructed=np.dot(UU, Vt[:k_components]) + mean_
    return(reconstructed)


def compute_Idiff_PCA_one_dataset_from_TC(data,trimming_length=0):
    '''
    Compute the Idiff values as a function of the number of components
    as proposed in the paper by Amico and Goni "The quest for identifiability
    in human functional connectomes"

    input: data -> dictionary in the form {subjID: fMRI data}
           trimming_length -> default 0 (no trimming), if different from 0 then trim
                              all the recordings to the same length

    output: numpy array containing the following rows:
            (k_components,Differential Identifiability score (%))
    '''

    nsubjects=len(data)
    list_FCs=computing_FCs_halved(data,trimming_length=trimming_length)
    all_connectomes=list_FCs.reshape((nsubjects*2,len(list_FCs[0][0])))
    N_total=len(all_connectomes)
    mean_ = np.mean(all_connectomes, axis=0)
    all_connectomes -=mean_
    U, S, Vt= np.linalg.svd(all_connectomes, full_matrices=False)
    U, Vt = svd_flip(U, Vt)
    # pca = PCA(n_components=N_total)
    # U, S, Vt=pca._fit(all_connectomes)

    ## Doing the PCA reconstruction keeping k components
    list_Idiff=np.array([(k_components,
      compute_Imat_from_listFCs(fit_kcomponents(U,S,Vt,all_connectomes,mean_,k_components))[1])
      for k_components in range(1,len(all_connectomes)+1)])

#    #The previous lines are equivalent to these
#     list_Idiff=[]    
#     for k_components in range(1,len(all_connectomes)+1):
#         reconstructed_connectomes=fit_kcomponents(pca,U,S,Vt,all_connectomes,k_components)
#         _,current_Idiff,_,_=compute_Imat_from_listFCs(reconstructed_connectomes)
#         list_Idiff.append([k_components,current_Idiff])
#     list_Idiff=np.array(list_Idiff)

    return(list_Idiff)


def compute_Idiff_PCA_one_dataset_from_FCs(list_FCs):
    '''
    Compute the Idiff values as a function of the number of components
    as proposed in the paper by Amico and Goni "The quest for identifiability
    in human functional connectomes"

    input: list_FCs -> a 3D array of size (nsubjects x 2 x nedges) reporting the 
                       flattened upper triangular matrix of functional connectomes 
                       constructed when splitting each fMRI recording in two halves

    output: numpy array containing the following rows:
            (k_components,Differential Identifiability score (%))
    '''

    nsubjects=len(list_FCs)
    all_connectomes=list_FCs.reshape((nsubjects*2,len(list_FCs[0][0])))
    N_total=len(all_connectomes)
    mean_ = np.mean(all_connectomes, axis=0)
    all_connectomes -=mean_
    U, S, Vt= np.linalg.svd(all_connectomes, full_matrices=False)
    U, Vt = svd_flip(U, Vt)
    # pca = PCA(n_components=N_total)
    # U, S, Vt=pca._fit(all_connectomes)

    ## Doing the PCA reconstruction keeping k components
    list_Idiff=np.array([(k_components,
      compute_Imat_from_listFCs(fit_kcomponents(U,S,Vt,all_connectomes,mean_,k_components))[1])
      for k_components in range(1,len(all_connectomes)+1)])

    return(list_Idiff)



def compute_Idiff_PCA_two_datasets_from_TC(data1,data2,trimming_length=0):
    '''
    Compute the Idiff values as a function of the number of components
    as proposed in the paper by Amico and Goni "The quest for identifiability
    in human functional connectomes"

    input: data1 -> dictionary in the form {subjID: fMRI data} %% TEST
           data2 -> dictionary in the form {subjID: fMRI data} %% RETEST
           trimming_length -> default 0 (no trimming), if different from 0 then trim
                              all the recordings to the same length

    output: numpy array containing the following rows:
            (k_components,Differential Identifiability score (%))

    '''

    nsubjects=len(data1)
    n_ROIs=np.shape(list(data1.values())[0])[0]
    u,v=np.triu_indices(n=n_ROIs,k=1,m=n_ROIs)
    FC1_list=computing_FCs(data1,trimming_length)
    FC2_list=computing_FCs(data2,trimming_length)
    ##Analogous command of np.array(list(zip(FC1_list,FC2_list)))) but slightly faster
    list_FCs=np.stack((FC1_list,FC2_list), axis=1)
    #list_FCs=computing_FCs_halved(data,trimming_length=trimming_length)
    all_connectomes=list_FCs.reshape((nsubjects*2,len(list_FCs[0][0])))
    N_total=len(all_connectomes)
    # pca = PCA(n_components=N_total)
    # U, S, Vt=pca._fit(all_connectomes)
    mean_ = np.mean(all_connectomes, axis=0)
    all_connectomes -=mean_
    U, S, Vt= np.linalg.svd(all_connectomes, full_matrices=False)
    U, Vt = svd_flip(U, Vt)

    ## Doing the PCA reconstruction keeping k components
    list_Idiff=np.array([(k_components,
      compute_Imat_from_listFCs(fit_kcomponents(U,S,Vt,all_connectomes,mean_,k_components))[1])
      for k_components in range(1,len(all_connectomes)+1)])

    return(list_Idiff)


def Idiff_from_FC_parallel(k_components):
    reconstructed_connectome=fit_kcomponents(U,S,Vt,list_connectomes,mean_,k_components)
    _,Idiff,_,_=compute_Imat_from_listFCs(reconstructed_connectome)
    return(k_components,Idiff)

def create_pca_approach_parallel(all_connectomes,N_total):
    global list_connectomes, U, S, Vt,mean_
    list_connectomes=all_connectomes
    mean_ = np.mean(list_connectomes, axis=0)
    list_connectomes -= mean_
    U, S, Vt= np.linalg.svd(list_connectomes, full_matrices=False)
    U, Vt = svd_flip(U, Vt)

def compute_Idiff_PCA_two_datasets_from_FCs(FC1_list,FC2_list,parallel=None):
    '''
    Compute the Idiff values as a function of the number of components
    as proposed in the paper by Amico and Goni "The quest for identifiability
    in human functional connectomes" from two lists of FCs

    input: FC1_list -> a 2D array of size (nsubjects x nedges) with flattened FCs for TEST
           FC2_list -> a 2D array of size (nsubjects x nedges) with flattened FCs for RETEST

    output: numpy array containing the following rows:
            (k_components,Differential Identifiability score (%))
            
    '''

    nsubjects=len(FC1_list)
    ##Analogous command of np.array(list(zip(FC1_list,FC2_list)))) but slightly faster
    list_FCs=np.stack((FC1_list,FC2_list), axis=1)
    all_connectomes=list_FCs.reshape((nsubjects*2,len(FC1_list[0])))
    N_total=len(all_connectomes)
    if parallel==None:
        #Doing the PCA
        mean_ = np.mean(all_connectomes, axis=0)
        all_connectomes -=mean_
        U, S, Vt= np.linalg.svd(all_connectomes, full_matrices=False)
        U, Vt = svd_flip(U, Vt)
        ## Doing the PCA reconstruction keeping k components
        list_Idiff=np.array([(k_components,
          compute_Imat_from_listFCs(fit_kcomponents(U,S,Vt,all_connectomes,mean_,k_components))[1])
          for k_components in range(2,len(all_connectomes)+1)])
    else:
        ###Parallel processing is slower than sequential (due to too )
        results=[]
        pool = Pool(processes=parallel, initializer=create_pca_approach_parallel,
                initargs=(all_connectomes, N_total))
        
        for k_components in range(1,len(all_connectomes)+1):
            results.append(pool.apply_async(Idiff_from_FC_parallel,args=(k_components,)))
        pool.close()
        pool.join()
        list_Idiff=np.array(sorted([list(i.get()) for i in results],key=lambda x: x[0],reverse=False))

    return(list_Idiff)

def compute_FCs_PCA_opt_two_datasets_from_FCs(FC1_list,FC2_list,m_star):
    '''
    Compute reconstructed Functional Connectome with highest Idiff 
    as proposed in the paper by Amico and Goni "The quest for identifiability
    in human functional connectomes" from two lists of FCs

    input: FC1_list -> a 2D array of size (nsubjects x nedges) with flattened FCs for TEST
           FC2_list -> a 2D array of size (nsubjects x nedges) with flattened FCs for RETEST
           m_star -> number of PCA components with highest Idiff

    output: 
            FC1_list_opt -> reconstructed FC1_list with highest Idiff 
            FC2_list_opt -> reconstructed FC2_list with highest Idiff
    '''
    nsubjects=len(FC1_list)
    ##Analogous command of np.array(list(zip(FC1_list,FC2_list)))) but slightly faster
    list_FCs=np.stack((FC1_list,FC2_list), axis=1)
    all_connectomes=list_FCs.reshape((nsubjects*2,len(FC1_list[0])))
    N_total=len(all_connectomes)
    mean_ = np.mean(all_connectomes, axis=0)
    all_connectomes -=mean_
    U, S, Vt= np.linalg.svd(all_connectomes, full_matrices=False)
    U, Vt = svd_flip(U, Vt)
    all_connectomes_opt = fit_kcomponents(U,S,Vt,all_connectomes,mean_,m_star)
    FC1_list_opt = all_connectomes_opt[0::2]
    FC2_list_opt = all_connectomes_opt[1::2]
    return(FC1_list_opt, FC2_list_opt)

def compute_Imat_PCA_opt_two_datasets_from_FCs(FC1_list,FC2_list,list_Idiff):
    '''
    Compute Imat, Idiff, Iself, Iothers with highest Idiff 
    as proposed in the paper by Amico and Goni "The quest for identifiability
    in human functional connectomes" from two lists of FCs

    input: FC1_list -> a 2D array of size (nsubjects x nedges) with flattened FCs for TEST
           FC2_list -> a 2D array of size (nsubjects x nedges) with flattened FCs for RETEST

    output: 
            Imat_opt -> Identifiability matrix with largest Idiff
            Idiff_opt -> largest Differential Identifiability score (%)
            Iself_opt
            Iothers_opt
    '''

    nsubjects=len(FC1_list)
    ##Analogous command of np.array(list(zip(FC1_list,FC2_list)))) but slightly faster
    list_FCs=np.stack((FC1_list,FC2_list), axis=1)
    all_connectomes=list_FCs.reshape((nsubjects*2,len(FC1_list[0])))
    N_total=len(all_connectomes)
    mean_ = np.mean(all_connectomes, axis=0)
    all_connectomes -=mean_
    U, S, Vt= np.linalg.svd(all_connectomes, full_matrices=False)
    U, Vt = svd_flip(U, Vt)

    Idiff_opt = np.max(list_Idiff[:,1])
    m_star = int(list_Idiff[np.where(list_Idiff[:,1] == Idiff_opt)[0][0], 0])
    
    Imat_opt,Idiff_opt,Iself_opt,Iothers_opt = compute_Imat_from_listFCs(fit_kcomponents(U,S,Vt,all_connectomes,mean_,m_star))
    return(Imat_opt, Idiff_opt, Iself_opt, Iothers_opt)

def plot_classic_Imat_and_PCAcurve(data_test,data_retest=None,trimming_length=0,vmin=-0.2,vmax=1,condition='Rest',parallel=None,
                                   ticks_width=1.5, axis_width=2.5, labelsizereg=16):
    '''
    Function that plot Imat and the Idiff values as a function of the number of components
    as proposed in the paper by Amico and Goni "The quest for identifiability
    in human functional connectomes".
    If only one dataset is present, then do the procedure when splitting the recordings in half

    input: data1 -> dictionary in the form {subjID: fMRI data} %% TEST
           data2 -> dictionary in the form {subjID: fMRI data} %% RETEST
           trimming_length -> default 0 (no trimming), if different from 0 th
    '''
    if data_retest==None:
        FC_list=computing_FCs_halved(data_test,trimming_length=trimming_length)
        I_mat,Idiff,Iself,Iothers=compute_Imat_from_single_session_FCs(FC_list)
    else:
        FC1_list=computing_FCs(data_test,trimming_length=trimming_length)
        FC2_list=computing_FCs(data_retest,trimming_length=trimming_length)
        I_mat,Idiff,Iself,Iothers=compute_Imat_from_FCs_twodatasets(FC1_list,FC2_list)
    fig = plt.figure(dpi=150,figsize=(12,6))
    ax= plt.subplot(121)
    
    im=plt.imshow(I_mat,vmin=vmin,vmax=vmax)
    cbar=plt.colorbar(im,fraction=0.046, pad=0.04)
    if data_retest==None:
        plt.xlabel('Subjects Test (first half)')
        plt.ylabel('Subjects Retest (second half)')
    else:
        plt.xlabel('Subjects Test (day 1)')
        plt.ylabel('Subjects Retest (day 2)')
    cbar.set_label(r"Pearson's $\rho$", rotation=90,labelpad=-5)
    num_title=list([np.round(i,2) for i in [Idiff,Iself,Iothers]])
    plt.title('%s \n Idiff: %.2f %%, Iself: %.2f, Iothers: %.2f' %(condition,num_title[0],num_title[1],num_title[2]));
    plot_adjustments(ax,ticks_width=ticks_width, axis_width=axis_width, labelsizereg=labelsizereg)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)


    ax= plt.subplot(122)

    if data_retest==None:
        PCA_list_idiff=compute_Idiff_PCA_one_dataset_from_FCs(FC_list)
    else:
        PCA_list_idiff=compute_Idiff_PCA_two_datasets_from_FCs(FC1_list,FC2_list,parallel=parallel)
    ax.plot(PCA_list_idiff[:,0],PCA_list_idiff[:,1],'ro-')
    asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
    ax.set_aspect(asp)
    plt.xlabel('Number of PCA components')
    plt.ylabel(r'$I_{diff}$ (%)',fontsize=14,labelpad=-1)
    plt.subplots_adjust(wspace=0.35)
    plot_adjustments(ax,ticks_width=ticks_width, axis_width=axis_width, labelsizereg=labelsizereg)
    return(I_mat,PCA_list_idiff)




## ICC functions
    

def compute_ICC_parallel(i,mat):
    return(i,ICC(mat)[0])
    

def compute_ICC_idenfiability_from_FCs(FC_lists,parallel=None):
    '''
    Compute the ICC matrix '1-1' considering the list of flattened functional connectomes
    '''
    
    num_FCs=len(FC_lists)
    nsubjs,edges=FC_lists[0].shape
    nROIs=int(np.ceil(np.sqrt(2*edges)))
    ICC_matrix=np.zeros((nROIs,nROIs))
    u,v= np.triu_indices(n=nROIs,k=1,m=nROIs)
    ICC_values=np.zeros((edges))
    if parallel==None:
        for i in range(edges):
            current_mat_edges=np.vstack([FC_lists[s][:,i] for s in range(num_FCs)])
            
            ICC_values[i]=ICC(current_mat_edges.T)[0]
    else:
        pool=Pool(processes=parallel)
        res=[]
        for i in range(edges):
            
            ICC_mat=np.vstack([FC_lists[s][:,i] for s in range(num_FCs)]).T
            res.append(
                pool.apply_async(compute_ICC_parallel,
                                 args=(i,ICC_mat)))
        pool.close()
        pool.join()
        list_ICC=np.array(sorted([list(i.get()) for i in res],key=lambda x: x[0],reverse=False))
#        print(list_ICC)
        ICC_values=list_ICC[:,1]
    ICC_matrix[u,v]=ICC_values
    ICC_matrix+=ICC_matrix.T
    return(ICC_matrix)
    
def compute_ICC_idenfiability_from_single_FCs(FC_list,parallel=None):
    '''
    Compute the ICC matrix '1-1' considering the list of flattened functional connectomes
    '''
    
    FC1_list=FC_list[:,0,:]
    FC2_list=FC_list[:,1,:]
    
    nsubjs,edges=FC1_list.shape
    nROIs=int(np.ceil(np.sqrt(2*edges)))
    ICC_matrix=np.zeros((nROIs,nROIs))
    u,v= np.triu_indices(n=nROIs,k=1,m=nROIs)
    
    ICC_values=np.zeros((edges))
    if parallel==None:
        for i in range(edges):
            ICC_values[i]=ICC(np.vstack((FC1_list[:,i],FC2_list[:,i])).T)[0]
    else:
        pool=Pool(processes=parallel)
        res=[]
        for i in range(edges):
            ICC_mat=np.vstack( (FC1_list[:,i],FC2_list[:,i]) ).T
            res.append(
                pool.apply_async(compute_ICC_parallel,
                                 args=(i,ICC_mat)))
        pool.close()
        pool.join()
        list_ICC=np.array(sorted([list(i.get()) for i in res],key=lambda x: x[0],reverse=False))
        ICC_values=list_ICC[:,1]
    ICC_matrix[u,v]=ICC_values
    ICC_matrix+=ICC_matrix.T
    return(ICC_matrix)




def find_limit_yeoOrder(yeovector,deflist=['VIS','SM','DA','VA','L','FP','DMN','C'],N=400,subcortical=None):
    scroll_deflist=0
    if subcortical!=None:
        list_yeo_ends={i:[0,0] for i in deflist}
    else:
        list_yeo_ends={i:[0,0] for i in deflist[:-1]}
    #print(list_yeo_ends)
    yeo_init=0
    for idx,i in enumerate(yeovector[:-1]):
        val_current=i
        val_prox=yeovector[idx+1]
        if val_prox-val_current < 0:
            yeo_end=idx
            list_yeo_ends[deflist[scroll_deflist]][0]=yeo_init
            list_yeo_ends[deflist[scroll_deflist]][1]=yeo_end+1
            scroll_deflist+=1
            yeo_init=idx+1
    ##This is for the DMN
    list_yeo_ends[deflist[scroll_deflist]][0]=yeo_init
    list_yeo_ends[deflist[scroll_deflist]][1]=N
    if subcortical != None:
        list_yeo_ends[deflist[scroll_deflist+1]][0]=N+1
        list_yeo_ends[deflist[scroll_deflist+1]][1]=len(yeovector)
    return(list_yeo_ends) 



def load_yeonets_matfile(filename='/home/andrea/Dropbox/Brain_data/Misc/yeoOrder/yeo_RS7_Schaefer400S.mat',subcortical=None):
    Yeonets=loadmat(filename)
    N_yeos=len(Yeonets['yeoOrder'])
    if subcortical==None:
        yeoOrder=np.array([i[0]-1 for i in Yeonets['yeoOrder']])[:int(N_yeos-19)]
        limit_yeo=find_limit_yeoOrder(yeoOrder,N=int(N_yeos-19),subcortical=None)
    else:
        yeoOrder=np.array([i[0]-1 for i in Yeonets['yeoOrder']])
        limit_yeo=find_limit_yeoOrder(yeoOrder,N=N_yeos,subcortical=True)
    return(yeoOrder,limit_yeo)


def plot_ICC_mat(ICC_mat,yeo_net=True,yeoOrder=None,limit_yeo=None,
                subcortical=None,percentile=95,ax=None):
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
        ax.matshow(ICC_mat,cmap=cmap)
    else:
        ICC_mat_reodered=ICC_mat[yeoOrder,:][:,yeoOrder]
        yeo_ICC_mat=np.where(ICC_mat_reodered>np.percentile(ICC_mat,q=percentile),0,np.nan)
        k=1
        for net,limits in limit_yeo.items():
            y0,y1=limits[0],limits[1]
#             print(y0,y1,k)
            yeo_ICC_mat[y0:y1,y0:y1]=np.where(yeo_ICC_mat[y0:y1,y0:y1]==0,k,np.nan)
            k+=1

        ax.matshow(yeo_ICC_mat,cmap=cmap)
        ax.set_xlim(-0,len(ICC_mat));
        ax.set_ylim(len(ICC_mat),0);
        s=0
        net_names=['VIS','SM','DA','VA','L','FP','DMN','SC']
        for net,limits in limit_yeo.items():
            y0,y1=limits[0],limits[1]
#             if y0 <= len(ICC_mat) and y1<= len(ICC_mat):
                #print(y0,y1)
            rect_upper = Rectangle((len(ICC_mat)+2,y0),5, y1-y0, color =list_colors_yeo_center[s+1], clip_on=False,zorder=10)
            rect_bottom = Rectangle((y0,len(ICC_mat)+2),y1-y0,5,color =list_colors_yeo_center[s+1], clip_on=False,zorder=10)
            ax.text(x=1.025*len(ICC_mat),y=y0+(0.65*(y1-y0)),s=net_names[s],clip_on=False)
            s+=1
            ax.add_patch(rect_upper)
            ax.add_patch(rect_bottom)
            plt.hlines(y0,y0,y1,colors='k',lw=1)
            plt.hlines(y1,y0,y1,colors='k',lw=1)
            plt.vlines(y0,y0,y1,colors='k',lw=1)
            plt.vlines(y1,y0,y1,colors='k',lw=1)



def compute_ICC_yeo_net_dataframe(ICC_mat,yeoOrder,limit_yeo):
    ICC_mat_reodered=ICC_mat[yeoOrder,:][:,yeoOrder]
    list_average_ICC=[]
    list_yeo_repeated=[]
    s=1
    for net,limits in limit_yeo.items():
        y0,y1=limits[0],limits[1]
        gap=y1-y0
#         if y0 <= len(ICC_mat) and y1<= len(ICC_mat):
#         print(y0,y1,len(ICC_mat_reodered[y0:y1,y0:y1][~np.eye(gap,dtype=bool)].flatten()),(gap)*(gap-1))
        list_average_ICC.extend(ICC_mat_reodered[y0:y1,y0:y1][~np.eye(gap,dtype=bool)].flatten())
        list_yeo_repeated.extend(np.repeat(s,(gap)*(gap-1)))
        s+=1
    current_df=pd.DataFrame([list_average_ICC,list_yeo_repeated]).T
    current_df=current_df.rename(columns={0:'ICC',1:'YeoNet'})
    return(current_df)

def plot_ICC_violins(ICC_mat,yeoOrder,limit_yeo,ax=None):
    current_df=compute_ICC_yeo_net_dataframe(ICC_mat,yeoOrder,limit_yeo)
    list_colors_yeo=['#969696','#7fc97f', '#beaed4',
                     '#fdc086','#ffeda0', '#386cb0',
                     '#f0027f','#bf5b17']
    if ax == None:
        fig=plt.figure(dpi=150)
        ax=plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sns.violinplot(x=current_df['YeoNet'],y=current_df['ICC'],orient='v',cut=0.5,
                   palette=sns.color_palette(list_colors_yeo[1:]),alpha=0.5,width=0.7,linewidth=1)
    
    net_names=list(limit_yeo.keys())
    # Set the x labels
    ax.set_ylabel('ICC Values',fontsize=14)
    ax.set_xlabel('')
    ax.set_ylim(0,1)
    ax.set_xticklabels(net_names);


def plot_ICC_mat_and_violins(ICC_mat,yeoOrder,limit_yeo,percentile=95,subcortical=None):
    fig=plt.figure(figsize=(10,6),dpi=150)
    gs = GridSpec(6, 2)
    ax=fig.add_subplot(gs[0:5, 0])
    plot_ICC_mat(ICC_mat,yeo_net=True,yeoOrder=yeoOrder,subcortical=subcortical,
                limit_yeo=limit_yeo,percentile=percentile,ax=ax)

    ax=fig.add_subplot(gs[1:4, 1])
    plot_ICC_violins(ICC_mat,yeoOrder=yeoOrder,limit_yeo=limit_yeo,ax=ax)
    plt.subplots_adjust(wspace=0.32)

def save_ICC_mat_and_violins(save_path, ICC_mat,yeoOrder,limit_yeo,percentile=95,subcortical=None):
    fig=plt.figure(figsize=(10,6),dpi=150)
    gs = GridSpec(6, 2)
    ax=fig.add_subplot(gs[0:5, 0])
    plot_ICC_mat(ICC_mat,yeo_net=True,yeoOrder=yeoOrder,subcortical=subcortical,
                limit_yeo=limit_yeo,percentile=percentile,ax=ax)

    ax=fig.add_subplot(gs[1:4, 1])
    plot_ICC_violins(ICC_mat,yeoOrder=yeoOrder,limit_yeo=limit_yeo,ax=ax)
    plt.subplots_adjust(wspace=0.32)

    plt.savefig(save_path)
    plt.close()
