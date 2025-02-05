from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

from . import map_func
from . import stat_func


def convert_to_array(df, subj_name, var_name, tgt_name, sep_name):
    """
    Convert dataframe to numpy array.
    ---------------------------------------------------------------------------
    Parameters:
    ---------------------------------------------------------------------------
    df (pd.DataFrame): The input dataframe.
    subj_name (str): Subject identifier column name.
    var_name (str): Variable column name.
    tgt_name (str): Variable column name that map together.
    sep_name (str): Variable column name that map separately. Separate in array.
    ---------------------------------------------------------------------------
    """
    subjs = np.sort(df[subj_name].unique())
    seps = np.sort(df[sep_name].unique())

    n_subjs = len(subjs)
    n_vars = len(var_name)
    n_seps = len(seps)
    n_imgs = 0

    for i, cond in enumerate(seps):
        cond_data = df[df[sep_name] == cond]
        imgs = cond_data[tgt_name].unique()
        cond_n_imgs = len(imgs)
        if cond_n_imgs > n_imgs:
            n_imgs = cond_n_imgs

    output = np.zeros((n_seps, n_vars, n_subjs, n_imgs))

    for i, cond in enumerate(seps):
        cond_data = df[df[sep_name] == cond]
        imgs = np.sort(cond_data[tgt_name].unique())
        for j, metric in enumerate(var_name):
            for k, subj in enumerate(subjs):
                subj_data = cond_data[cond_data[subj_name] == subj]
                for m, img in enumerate(imgs):
                    if img in subj_data[tgt_name].values:
                        img_data = subj_data[subj_data[tgt_name] == img]
                        output[i,j,k,m] = img_data[metric].values[0]
                    else:
                        output[i,j,k,m] = np.nan
            imputer = SimpleImputer(strategy='mean')
            output[i,j,:,:] = imputer.fit_transform(output[i,j,:,:])

    return output


def split_arr(human, model, n_bs, seed=42):
    """
    Split array into train and test sets.
    ---------------------------------------------------------------------------
    Parameters:
    ---------------------------------------------------------------------------
    human (np.ndarray): The human data array.
    model (np.ndarray): The model data array.
    n_bs (int): Number of bootstrap samples.
    seed (int): Random seed for reproducibility.
    ---------------------------------------------------------------------------
    """

    np.random.seed(seed)
    img_axis = human.shape[-1]
    all_indices = np.arange(img_axis)
    out_human = np.zeros((n_bs, 2, human.shape[0], human.shape[1], human.shape[2], int(human.shape[-1]/2)))
    out_model = np.zeros((n_bs, 2, model.shape[0], model.shape[1], model.shape[2], int(model.shape[-1]/2)))

    for i in range(n_bs):
        chosen = np.random.choice(img_axis, int(img_axis/2), replace=False)
        unchosen = np.setdiff1d(all_indices, chosen)

        out_human[i, 0, :, :, :, :] = human[:, :, :, chosen]
        out_human[i, 1, :, :, :, :] = human[:, :, :, unchosen]
        out_model[i, 0, :, :, :, :] = model[:, :, :, chosen]
        out_model[i, 1, :, :, :, :] = model[:, :, :, unchosen]

    return out_human, out_model


def compute_full_corr_matrix(arr1, arr2):
    """
    Compute the full correlation matrix between two arrays.
    ---------------------------------------------------------------------------
    Parameters:
    ---------------------------------------------------------------------------
    arr1 (np.ndarray): The first input array.
    arr2 (np.ndarray): The second input array.
    ---------------------------------------------------------------------------
    """

    # normalize both array and compute dot product
    arr1 = (arr1 - np.nanmean(arr1, axis = 1, keepdims=True)) / np.nanstd(arr1, axis = 1, keepdims=True)
    arr2 = (arr2 - np.nanmean(arr2, axis = 1, keepdims=True)) / np.nanstd(arr2, axis = 1, keepdims=True)
    corr_matrix = np.dot(arr1, arr2.T) / (arr1.shape[-1])
    return corr_matrix


def mapping_matrix(arr1, arr2):
    """
    Compute the full correlation matrix between two sets of raw data.
    ---------------------------------------------------------------------------
    Parameters:
    arr1 (np.ndarray): The first input array.
    arr2 (np.ndarray): The second input array.
    ---------------------------------------------------------------------------
    For both array, the axes refers to:
    0. Bootstrap sample
    1. Bootstrap split
    2. Experimental Condition (map_sep)
    3. Variable (map_var)
    4. Subject
    5. Image
    """

    # check arr1 and arr2 to see if they are exactly the same
    same = np.array_equal(arr1, arr2)

    assert arr1.shape == arr2.shape, "Shape mismatch between the two arrays in mapping matrix"

    if same:
        output = np.zeros((arr1.shape[0], arr1.shape[1], arr1.shape[2], arr1.shape[3], arr1.shape[4], arr1.shape[4] - 1))
    else:
        output = np.zeros((arr1.shape[0], arr1.shape[1], arr1.shape[2], arr1.shape[3], arr1.shape[4], arr1.shape[4]))

    for i in range(arr1.shape[0]):
        for j in range(arr1.shape[1]):
            for k in range(arr1.shape[2]):
                for l in range(arr1.shape[3]):
                    result = compute_full_corr_matrix(arr1[i, j, k, l], arr2[i, j, k, l])

                    if same:
                        np.fill_diagonal(result, np.nan)
                        result = result[~np.isnan(result)]
                        result = result.reshape(arr1.shape[4], arr1.shape[4]-1)

                    output[i,j,k,l,:,:] = result

    output = stat_func.r2z(output, 'pearson')
    output = np.mean(output, axis=2)
    output = stat_func.z2r(output, 'pearson')
    return output


def retain_max_per_row_in_mat(arr):
    """
    Simple function to take in a 2D array and return a 2D array
    with only the max value per row, else nan
    """

    # Copy the data to avoid modifying the original array
    max_only = np.full_like(arr, np.nan)
    # Iterate over each row
    for row_idx in range(arr.shape[0]):
        # Find the index of the maximum value in the row
        for i in range(1, 2):
            max_col_idx = np.argsort(arr[row_idx, :])[-i]
            max_only[row_idx, max_col_idx] = arr[row_idx, max_col_idx]
    return max_only


def shuffle_image_order(arr, seed=42):
    """
    Shuffle the order of images in the last axis of the array.
    ---------------------------------------------------------------------------
    Parameters:
    ---------------------------------------------------------------------------
    arr (np.ndarray): The input array.
    seed (int): Random seed for reproducibility.
    ---------------------------------------------------------------------------
    """

    np.random.seed(seed)
    shuf_arr = np.copy(arr)
    shuf_arr = shuf_arr[..., np.random.permutation(shuf_arr.shape[-1])]
    return shuf_arr


def center_to_zero(arr):
    """
    Center the array to zero in the last axis of the array (images).
    ---------------------------------------------------------------------------
    Parameters:
    ---------------------------------------------------------------------------
    arr (np.ndarray): The input array.
    ---------------------------------------------------------------------------
    """

    cent_arr = np.copy(arr)
    cent_arr = cent_arr - np.mean(cent_arr, axis = -1, keepdims=True)
    return cent_arr





