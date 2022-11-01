# Author: Amish Mishra
# Date: April 27, 2022
# README: This file generates all Persistence statistics for the input data after performing a delay embedding
# We caculate mean, standard deviation, skewness, kurtosis, 25th, 50th, 75th percentile, and the persistent entropy 
# for the set M and L where M = {(d+b)/2} and L = {d-b} with (b,d) \in PD

# import required module
import os
import pandas
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import cechmate as cm
from ripser import ripser
import time


def get_pd(filtration_method, data):
    if filtration_method.lower() == "alpha":
        alpha = cm.Alpha(verbose=False)
        filtration = alpha.build(2 * data)  # Alpha goes by radius instead of diameter
        dgms = alpha.diagrams(filtration, verbose=False)
    elif filtration_method.lower() == "rips":
        # only compute homology classes of dimension 1 less than the dimension of the data
        dgm_with_inf = ripser(data, maxdim=(len(data[0])-1))['dgms']
        dgms = dgm_with_inf
        dgms[0] = dgm_with_inf[0][:-1, :]   # remove the H_0 class with infinite persistence
    elif filtration_method.lower() == "del_rips":
        del_rips = cm.DR(verbose=False)
        filtration = del_rips.build(data)
        dgms = del_rips.diagrams(filtration, verbose=False)
    else:
        raise Exception(f'{filtration_method} is not a filtration name')
    return dgms


def plot(vector_list, title='Plot'):    # Helps to plot data in case we'd like to visualize
    d = len(vector_list[0])
    fig = plt.figure()
    plt.axis('equal')
    # Plot the graph if points are in less than 3 dimensions
    if (d >= 3):
        x = list()
        y = list()
        z = list()
        for v in vector_list:
            x.append(v[0])
            y.append(v[1])
            z.append(v[2])
        # 3D Plotting
        ax = plt.axes(projection="3d")
        ax.scatter(x, y, z)
    if (d == 2):
        x = list()
        y = list()
        for v in vector_list:
            x.append(v[0])
            y.append(v[1])
        plt.scatter(x, y)
    if (d == 1):
        x = list()
        y = np.zeros(len(vector_list))
        for v in vector_list:
            x.append(v[0])
        plt.scatter(x, y)
    plt.title(title)
    plt.show()


def delay_embedding(ts, dimen=2, delay=1, stride=1):
    """
    Performs a delay embedding of a d-dimensional time series into R^{dimen*d}
    Input
    -----
    time_series - (n,d) numpy array containing n time points of a d-dimensional time series
    dimen - integer >= 1 specifying the size of the embedding windows. Determines the embedding dimension.
    delay - integer >= 1 specifying the number of time points between consecutive components of the embedding space.
            delay = 1 (default) implies a continguous embedding window.
    stride - integer >= 1 the shift along the time series between the start points of consecutive windows.
    Returns
    -------
    point_cloud - (k, dimen*d) numpy array containing the k embedded points.
    """
    # ensure ts is given as a 2-dimensional array
    if len(ts.shape) < 2:
        ts = np.atleast_2d(ts).transpose()
    # determine the number of time points (n) and the dimension (d) of the time series
    n, d = ts.shape
    # size of the sliding window
    win_size = 1 + (dimen-1)*delay
    # number of points in the final point cloud
    num_pnts = int(np.floor((n - win_size) / stride)) + 1
    # loop over time series with sliding windows to construct delay embedding
    point_cloud = np.zeros( (num_pnts, d*dimen) )
    start = 0
    end = start + win_size
    i = 0
    while end <= n:
        point_cloud[i, :] = ts[start:end:delay,:].flatten()
        start = start + stride
        end = start + win_size
        i += 1
    return point_cloud


def persistent_entropy(arr, normalize=False):
    """
    Perform the persistent entropy values of a family of persistence barcodes (or persistence diagrams).
    Assumes that the input diagrams are from a determined dimension. If the infinity bars have any meaning
    in your experiment and you want to keep them, remember to give the value you desire to val_Inf.

    Parameters
    -----------
    normalize: bool, default False
        if False, the persistent entropy values are not normalized.
        if True, the persistent entropy values are normalized.
          
    Returns
    --------

    stat: float
        float of persistent entropy value

    """
    if all(arr > 0):
        total = np.sum(arr)
        p = arr / total
        stat = -np.sum(p * np.log(p))
        if normalize == True:
            stat = stat / np.log(len(arr))
    else:
        raise Exception("A bar is born after dying")

    return stat


def M_set(pd):
    M = pd.mean(axis=1)
    return M


def L_set(pd):
    L = pd[:, 1] - pd[:, 0]
    return L


def calculate_8_stats(arr):
    stat = np.zeros(8)
    if np.any(arr): # If the arr is non-empty, then compute the stats; otherwise, leave as 0's
        avg = np.mean(arr)
        std_dev = np.std(arr)
        skew = stats.skew(arr)
        kurtosis = stats.kurtosis(arr)
        percentile_25 = np.percentile(arr, 25)
        percentile_50 = np.percentile(arr, 50)
        percentile_75 = np.percentile(arr, 75)
        pers_entropy = persistent_entropy(arr, normalize=False)
        stat[0] = avg
        stat[1] = std_dev
        stat[2] = skew
        stat[3] = kurtosis
        stat[4] = percentile_25
        stat[5] = percentile_50
        stat[6] = percentile_75
        stat[7] = pers_entropy
    return stat


def generate_pers_stats(pds, keep_inf=False, val_inf=None):
    # Clear out an infinite classes if needed to
    if keep_inf == False:
        pds = [(dgm[dgm[:, 1] != np.inf]) for dgm in pds]
    if keep_inf == True:
        if val_inf != None:
            pds = [
                np.where(dgm == np.inf, val_inf, dgm)
                for dgm in pds
            ]
        else:
            raise Exception(
                "Remember: You need to provide a value to infinity bars if you want to keep them."
            )
    double_num_stats = 8*2
    pers_stat_matrix = np.full((len(pds), double_num_stats), np.nan)
    for i, k_pd in enumerate(pds):
        M = M_set(k_pd)
        L = L_set(k_pd)
        M_stats = calculate_8_stats(M)
        L_stats = calculate_8_stats(L)
        pers_stat_matrix[i] = np.concatenate([M_stats, L_stats])
    return pers_stat_matrix.flatten()     


def generate_pers_stats_table(directory, filtration_method, max_num_files, dimension=3, verbose=True):
    """
    Generates a data frame of all persistence diagrams' persistence statistics
    Input
    -----
    directory - string path to folder where csvs are contained with data
    max_num_files = integer specifying max number of files to run on (files are named in the order 1.csv, 2.csv, ...)
    Returns
    -------
    pers_stats_df = dataframe of all persistence statistics for each epoch
    """

    # initialize variables
    stages_dict = {11: 'wake', 12: 'rem', 13: 's1', 14: 's2', 15: 's3', 16: 's4'}
    stages_arr = ['rem', 'wake', 's1', 's2', 's3', 's4']
    pers_stats_arr = []

    for patient in range(1, max_num_files+1):
        filename = f'{directory}/{patient}.csv'
        print(filename)
        if os.path.isfile(filename):
            tic = time.time() # Track runtime if wanted
            print('Loading', filename)
            data = pandas.read_csv(filename, header=None).iloc[:, :]
            for c in data:
                if patient == 3 and c == 222:
                    continue    # skip this patient's 222 epoch due to problems with degenerate facets in del-triangulation
                # embed data one epoch at a time
                epoch_data = data[c][:120]
                curr_sleep_stage = stages_dict[data[c][120]]
                embedded_data = delay_embedding(epoch_data, dimen=dimension, delay=5, stride=1)
                # plot(embedded_data, title=curr_sleep_stage)
                dgm = get_pd(filtration_method, embedded_data)
                stats_vect = generate_pers_stats(
                    dgm, keep_inf=False, val_inf=None)
                pers_stats_arr.append(np.concatenate(([patient],[stages_arr.index(curr_sleep_stage)], stats_vect)))
            run_time = time.time()-tic
            if verbose: print('Runtime for processing this file:', run_time)
    df = pandas.DataFrame(pers_stats_arr)
    df.rename(columns={0: 'patient', 1: 'sleep_stage'}, inplace=True)
    print(df) if verbose else ''
    return df


def generate_training_validation_pers_stats(t, m):
    print(f'=========== Generating {t} data persistence stats ===========')
    print(f'---- Using {m} ----')
    dim = 3
    data_table = generate_pers_stats_table(directory=f'CGMH_preprocessed_data/{t}', filtration_method=m,
                                        max_num_files=90, dimension=dim, verbose=True)
    print(data_table)
    # data_table.to_pickle(f'persistence_statistics/{t.lower()}_embed_dim_{dim}_pers_stats_{m}.pkl')
    print(f'Finished making {t.lower()}_embed_dim_{dim}_pers_stats_{m}.pkl')

if __name__ == '__main__':
    types = ['Training', 'Validation']
    methods = ['rips', 'alpha', 'del_rips']
    for t in types:
        for m in methods:
            generate_training_validation_pers_stats(t, m)