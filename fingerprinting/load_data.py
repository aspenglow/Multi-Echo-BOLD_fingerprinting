import numpy as np
import os
from tqdm import tqdm
import random
from scipy import io

def load_data_split(data_path, subjects_num, echoes_total_num):
    loaded_data_path = "loaded_data"
    if os.path.exists(os.path.join(loaded_data_path, "orig_matrixs_retest.npy")):
        TCs_test = np.load(os.path.join(loaded_data_path, "TCs_test.npy"))
        TCs_retest = np.load(os.path.join(loaded_data_path, "TCs_retest.npy"))
        FCs_test = np.load(os.path.join(loaded_data_path, "FCs_test.npy"))
        FCs_retest = np.load(os.path.join(loaded_data_path, "FCs_retest.npy"))
        orig_matrixs_test = np.load(os.path.join(loaded_data_path, "orig_matrixs_test.npy"))
        orig_matrixs_retest = np.load(os.path.join(loaded_data_path, "orig_matrixs_retest.npy"))

        return TCs_test, TCs_retest, FCs_test, FCs_retest, orig_matrixs_test, orig_matrixs_retest 

    print("Loading time series from data path...")
    subjects_total_num = int(len(os.listdir(data_path)) / (echoes_total_num))
    data_lists = os.listdir(data_path)
    data_lists.sort()

    test_data = np.genfromtxt(fname=os.path.join(data_path, data_lists[0]), dtype='float32', delimiter=' ')
    FC_side_length = test_data.shape[0]

    # Initialization
    count = np.zeros(echoes_total_num, dtype=int)
    TC_length = test_data.shape[1]
    TCs_shape = (echoes_total_num, subjects_total_num, test_data.shape[0], int(TC_length/2))
    TCs_test = np.zeros(TCs_shape)
    TCs_retest = np.zeros(TCs_shape)
    FCs_shape = (echoes_total_num, subjects_total_num, FC_side_length, FC_side_length)
    FCs_test = np.zeros(FCs_shape)
    FCs_retest = np.zeros(FCs_shape)
    ICCs_shape = (echoes_total_num, FC_side_length, FC_side_length)
    
    orig_matrixs_shape = (echoes_total_num, int(FC_side_length*(FC_side_length-1)/2), subjects_total_num)
    orig_matrixs_test = np.zeros(orig_matrixs_shape)
    orig_matrixs_retest = np.zeros(orig_matrixs_shape)

    for file in tqdm(data_lists, desc='data', leave=False):
        is_opt_comb = "echo-" not in file
        subject_id = file[file.find("sub-")+4 : file.find("_task-")]
        if not is_opt_comb:
            echo_index = file[file.find("echo-")+5 : file.find("echo-")+6]
            echo_index = int(echo_index) - 1
        else: 
            echo_index = 4  # the last echo is for the optimal combination
        # Load time series data
        data = np.genfromtxt(fname=os.path.join(data_path, file), dtype='float32', delimiter=' ')
        TC_length = data.shape[1]
        TC_test = data[:, :int(TC_length/2)]
        TC_retest = data[:, int(TC_length/2):]
        # Calculate functional connectivity (FC) of a time series
        FC_test = np.corrcoef(TC_test)
        FC_retest = np.corrcoef(TC_retest)
        assert(np.sum(np.isnan(FC_test)) == 0) # make sure that all the FCs are valid.
        assert(np.sum(np.isnan(FC_retest)) == 0)
        assert(np.sum(np.isinf(FC_test)) == 0)
        assert(np.sum(np.isinf(FC_retest)) == 0)
        # Flatten low triangle of a FC matrix to a vector
        mask = np.tril(np.full((FC_test.shape[0], FC_test.shape[0]), True, dtype=bool), -1)  
        orig_column_test = FC_test[mask]
        orig_column_retest = FC_retest[mask]

        subject_index = count[echo_index]
        TCs_test[echo_index, subject_index] = TC_test
        TCs_retest[echo_index, subject_index] = TC_retest
        FCs_test[echo_index, subject_index] = FC_test
        FCs_retest[echo_index, subject_index] = FC_retest
        orig_matrixs_test[echo_index, :, subject_index] = orig_column_test 
        orig_matrixs_retest[echo_index, :, subject_index] = orig_column_retest
        count[echo_index] += 1
    
    # Randomly sample subjects_num subjects.
    if subjects_num < subjects_total_num:
        sample = random.sample(range(0, subjects_total_num), subjects_num)
        sample.sort()
        TCs_test = TCs_test[:, sample]
        TCs_retest = TCs_retest[:, sample]
        FCs_test = FCs_test[:, sample]
        FCs_retest = FCs_retest[:, sample]
        orig_matrixs_test = orig_matrixs_test[:, :, sample]
        orig_matrixs_retest = orig_matrixs_retest[:, :, sample]

    
    if not os.path.exists(loaded_data_path):
        os.mkdir(loaded_data_path)

    if not os.path.exists(os.path.join(loaded_data_path, "orig_matrixs_retest.npy")):
        np.save(os.path.join(loaded_data_path, "TCs_test.npy"), TCs_test)
        np.save(os.path.join(loaded_data_path, "TCs_retest.npy"), TCs_retest)
        np.save(os.path.join(loaded_data_path, "FCs_test.npy"), FCs_test)
        np.save(os.path.join(loaded_data_path, "FCs_retest.npy"), FCs_retest)
        np.save(os.path.join(loaded_data_path, "orig_matrixs_test.npy"), orig_matrixs_test)
        np.save(os.path.join(loaded_data_path, "orig_matrixs_retest.npy"), orig_matrixs_retest)

    return TCs_test, TCs_retest, FCs_test, FCs_retest, orig_matrixs_test, orig_matrixs_retest 

def load_data_nonsplit(data_path, subjects_num, echoes_total_num):
    print("Loading time series from data path...")
    subjects_total_num = int(len(os.listdir(data_path)) / (echoes_total_num))
    data_lists = os.listdir(data_path)
    data_lists.sort()

    test_data = np.genfromtxt(fname=os.path.join(data_path, data_lists[0]), dtype='float32', delimiter=' ')
    FC_side_length = test_data.shape[0]

    # Initialization
    count = np.zeros(echoes_total_num, dtype=int)
    FCs_shape = (echoes_total_num, subjects_total_num, FC_side_length, FC_side_length)
    FCs = np.zeros(FCs_shape)
    ICCs_shape = (echoes_total_num, FC_side_length, FC_side_length)
    
    orig_matrixs_shape = (echoes_total_num, int(FC_side_length*(FC_side_length-1)/2), subjects_total_num)
    orig_matrixs = np.zeros(orig_matrixs_shape)

    for file in tqdm(data_lists, desc='data', leave=False):
        is_opt_comb = "echo-" not in file
        subject_id = file[file.find("sub-")+4 : file.find("_task-")]
        if not is_opt_comb:
            echo_index = file[file.find("echo-")+5 : file.find("echo-")+6]
            echo_index = int(echo_index) - 1
        else: 
            echo_index = 4  # the last echo is for the optimal combination
        # Load time series data
        TS = np.genfromtxt(fname=os.path.join(data_path, file), dtype='float32', delimiter=' ')
        # Calculate functional connectivity (FC) of a time series
        FC = np.corrcoef(TS)
        assert(np.sum(np.isnan(FC)) == 0) # make sure that all the FCs are valid.
        assert(np.sum(np.isinf(FC)) == 0)
        # Flatten low triangle of a FC matrix to a vector
        mask = np.tril(np.full((FC.shape[0], FC.shape[0]), True, dtype=bool), -1)  
        orig_column = FC[mask]

        subject_index = count[echo_index]
        FCs[echo_index, subject_index] = FC
        orig_matrixs[echo_index, :, subject_index] = orig_column
        count[echo_index] += 1
    
    # Randomly sample subjects_num subjects.
    if subjects_num < subjects_total_num:
        sample = random.sample(range(0, subjects_total_num), subjects_num)
        sample.sort()
        FCs = FCs[:, sample]
        orig_matrixs = orig_matrixs[:, :, sample]

    return FCs, orig_matrixs 

    