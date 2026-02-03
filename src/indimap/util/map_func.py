from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

from . import stat_func


def append_confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    This will compute per row confusion matrix and append it to the dataframe
    """

    # infer categories from union stim/resp
    categories = sorted(set(df['stim'].dropna()) | set(df['resp'].dropna()))
    label_to_idx = {cat: i for i, cat in enumerate(categories)}
    n_types = len(categories)
    confuse_mats = np.zeros((len(df), n_types, n_types), dtype=int)

    valid = df['stim'].notna() & df['resp'].notna()
    stim_idx = df.loc[valid, 'stim'].map(label_to_idx).to_numpy()
    resp_idx = df.loc[valid, 'resp'].map(label_to_idx).to_numpy()
    row_idx = np.where(valid)[0]
    confuse_mats[row_idx, stim_idx, resp_idx] = 1
    df['confuse_mat'] = list(confuse_mats)
    # for i, row in df.iterrows():
    #     confusion_matrix = np.zeros((n_types, n_types))
    #     if not pd.isna(row['stim']) and not pd.isna(row['resp']):
    #         stim_idx = label_to_idx[row['stim']]
    #         resp_idx = label_to_idx[row['resp']]
    #         confusion_matrix[stim_idx, resp_idx] += 1
    #         df.at[i, 'confuse_mat'] = confusion_matrix
    return df


def convert_to_array(df: pd.DataFrame, 
                     subj_name: str, 
                     var_name: str,
                     tgt_name: str, 
                     sep_name: str,
                     compute_confusion: bool = False
                     ) -> np.ndarray:
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

    if compute_confusion:
        df = append_confusion_matrix(df)
        agg_dict = {col: 'mean' for col in df.select_dtypes(include='number').columns}
        agg_dict['confuse_mat'] = lambda x: sum(x)

        df = (
            df.groupby([subj_name, sep_name, tgt_name], as_index=False)
            .agg(agg_dict)
            .reset_index()
            )
    else:
        df = (
            df.groupby([subj_name, sep_name, tgt_name], as_index=False)
            .mean(numeric_only=True)
            .reset_index()
        )

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

    if compute_confusion:
        cm_index = var_name.index('confuse_mat')
        n_cm = df['confuse_mat'].values[0].size
        output = np.zeros((n_seps, n_vars, n_subjs, n_imgs, n_cm))
    else:
        output = np.zeros((n_seps, n_vars, n_subjs, n_imgs, 1))

    for i, cond in enumerate(seps):
        cond_data = df[df[sep_name] == cond]
        imgs = np.sort(cond_data[tgt_name].unique())
        for j, metric in enumerate(var_name):
            for k, subj in enumerate(subjs):
                subj_data = cond_data[cond_data[subj_name] == subj]
                for m, img in enumerate(imgs):
                    if img in subj_data[tgt_name].values:
                        img_data = subj_data[subj_data[tgt_name] == img]
                        if metric != 'confuse_mat':
                            output[i,j,k,m] = img_data[metric].values[0]
                        else:
                            cm = sum(img_data['confuse_mat'].values)
                            output[i,j,k,m] = cm.flatten()
                    else:
                        output[i,j,k,m] = np.nan
            if metric != 'confuse_mat':
                imputer = SimpleImputer(strategy='mean')
                output[i,j,:,:,0] = imputer.fit_transform(output[i,j,:,:,0])

    return output


def check_for_extreme(human: np.ndarray, model: np.ndarray) -> None:
    """
    Check for extremely high/low accuracy in the data.
    Raise warning or error if found.
    ---------------------------------------------------------------------------
    Parameters:
    ---------------------------------------------------------------------------
    human (np.ndarray): The human data array.
    model (np.ndarray): The model data array.
    ---------------------------------------------------------------------------
    """

    human_flag_crit = np.mean(human[:, 0, :, :], axis=-2)
    model_flag_crit = np.mean(model[:, 0, :, :], axis=-2)

    flag_human = np.where((human_flag_crit > 0.95) | (human_flag_crit < 0.05))[1]
    flag_model = np.where((model_flag_crit > 0.95) | (model_flag_crit < 0.05))[1]
    if flag_human.size > 0:
        print(f"Warning: Human {flag_human} is achieving 95% or 5% accuracy. This may cause problem in the bootstrapping analysis.")
    if flag_model.size > 0:
        print(f"Warning: Instances {flag_model} is achieving 95% or 5% accuracy. This may cause problem in the bootstrapping analysis.")

    extreme_human = np.where((human_flag_crit > 0.99) | (human_flag_crit < 0.01))[1]
    extreme_model = np.where((model_flag_crit > 0.99) | (model_flag_crit < 0.01))[1]
    if extreme_human.size > 0:
        raise ValueError(f"Human {extreme_human} is achieving 99% or 1% accuracy, remove this subject")
    if extreme_model.size > 0:
        raise ValueError(f"Instances {extreme_model} is achieving 99% or 1% accuracy, remove this instance")


def random_split_half(human: np.ndarray, model: np.ndarray) -> tuple:
    """
    Recursive function to split the data into two halves.
    Ensure that no split contains only one unique value (failed to correlate).
    If so, resplit.
    ---------------------------------------------------------------------------
    Parameters:
    ---------------------------------------------------------------------------
    human (np.ndarray): The human data array (cond, metrics, subj, image, confusion).
    model (np.ndarray): The model data array.
    img (int): Image index to split.
    ---------------------------------------------------------------------------
    """

    resplit = False
    img_axis = human.shape[-2]
    all_indices = np.arange(img_axis)

    chosen = np.random.choice(img_axis, int(img_axis/2), replace=False)
    unchosen = np.setdiff1d(all_indices, chosen)
    for i in range(human.shape[0]):
        for k in range(human.shape[2]):
            check_split = [
                len(np.unique(human[i,0,k,chosen,0])),
                len(np.unique(human[i,0,k,unchosen,0])),
                len(np.unique(model[i,0,k,chosen,0])),
                len(np.unique(model[i,0,k,unchosen,0])),
            ]
            if 1 in check_split:
                print('Resplit')
                resplit = True
                break

    if resplit:
        return random_split_half(human, model)
    else:
        return chosen, unchosen


def stratified_split_half(human: np.ndarray, model: np.ndarray) -> tuple:
    """
    Split the last axis of human and model arrays in blocks of 40,
    randomly selecting 20 indices per block, while ensuring no degenerate splits.

    Parameters
    ----------
    human : np.ndarray
        Human data array, shape (..., n_images)
    model : np.ndarray
        Model data array, shape (..., n_images)

    Returns
    -------
    tuple
        Two arrays of indices: (chosen, unchosen)
    """
    n_images = human.shape[-1]
    block_size = 40
    select_per_block = 20
    chosen = []

    for start in range(0, n_images, block_size):
        block_indices = np.arange(start, min(start + block_size, n_images))
        # randomly pick 20 from this block
        chosen_block = np.random.choice(block_indices, size=select_per_block, replace=False)
        chosen.extend(chosen_block)

    chosen = np.array(chosen)
    unchosen = np.setdiff1d(np.arange(n_images), chosen)

    # Optional: check for degenerate splits (like before)
    resplit = False
    for i in range(human.shape[0]):
        for j in range(human.shape[1]):
            for k in range(human.shape[2]):
                check_split = [
                    len(np.unique(human[i,j,k,chosen])),
                    len(np.unique(human[i,j,k,unchosen])),
                    len(np.unique(model[i,j,k,chosen])),
                    len(np.unique(model[i,j,k,unchosen])),
                ]
                if 1 in check_split:
                    resplit = True
                    break
            if resplit:
                break
        if resplit:
            break

    if resplit:
        return stratified_split_half(human, model)
    else:
        return chosen, unchosen


def split_image_array(arr1: np.ndarray, 
                      arr2: np.ndarray, 
                      n_bs: int, 
                      seed: int = 42,
                      ) -> tuple:
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
    img_axis = arr1.shape[-2]
    all_indices = np.arange(img_axis)
    
    for _ in range(n_bs):
        chosen, unchosen = random_split_half(arr1, arr2)
        yield chosen, unchosen


def split_subj(n_subjs: int, 
               n_bs: int,
               seed: int = 42) -> np.ndarray:
    """
    Split subjects into two halves.
    ---------------------------------------------------------------------------
    Parameters:
    ---------------------------------------------------------------------------
    n_subjs (int): Number of subjects.
    seed (int): Random seed for reproducibility.
    ---------------------------------------------------------------------------
    Output: np.ndarray (shape: (n_bs, 2, half_subjs))
    """

    np.random.seed(seed)
    all_indices = np.arange(n_subjs)
    half_subjs = int(n_subjs / 2)
    output_indices = np.empty((n_bs, 2, half_subjs), dtype=int)
    for i in range(n_bs):
        bs_indices = np.random.permutation(all_indices)
        output_indices[i, 0, :] = bs_indices[:half_subjs]
        output_indices[i, 1, :] = bs_indices[half_subjs:]
    return output_indices


def compute_full_corr_matrix(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
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


def compute_full_corr_confusion(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    Compute the full correlation matrix between two confusion matrices.
    ---------------------------------------------------------------------------
    Parameters:
    ---------------------------------------------------------------------------
    arr1 (np.ndarray): The first input array. (subj, image, confusion matrix)
    arr2 (np.ndarray): The second input array (subj, image, confusion matrix).
    ---------------------------------------------------------------------------
    """

    # sum across images and normalize to percentage before computing full corr matrix
    arr1 = np.nansum(arr1, axis = 1) / np.nansum(arr1, axis = (1,2))[:, None]
    arr2 = np.nansum(arr2, axis = 1) / np.nansum(arr2, axis = (1,2))[:, None]
    corr_matrix = compute_full_corr_matrix(arr1, arr2)
    return corr_matrix


def mapping_matrix(arr1: np.ndarray, 
                   arr2: np.ndarray, 
                   map_var: list, 
                   same: bool,
                   sep_conds: bool = False,
                   n_bs: int = 1,
                   seed: int = 42,
                   ) -> np.ndarray:
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

    if 'confuse_mat' in map_var:
        cm_idx = map_var.index('confuse_mat')
    else:
        cm_idx = -1

    # assert arr1.shape == arr2.shape, "Shape mismatch between the two arrays in mapping matrix"

    if same:
        output = np.zeros((n_bs, 2,
                           arr1.shape[0], arr1.shape[1], 
                           arr1.shape[2], arr1.shape[2] - 1
                           ))
    else:
        output = np.zeros((n_bs, 2,
                           arr1.shape[0], arr1.shape[1], 
                           arr1.shape[2], arr2.shape[2]
                           ))

    for i, (split1, split2) in enumerate(split_image_array(arr1, arr2, n_bs, seed)):
        for j, idx in enumerate([split1, split2]):
            for k in range(arr1.shape[0]):
                for l in range(arr1.shape[1]):
                    if l == cm_idx:
                        result = compute_full_corr_confusion(
                            np.take(arr1[k, l, :, :, :], idx, axis = 1),
                            np.take(arr2[k, l, :, :, :], idx, axis = 1),
                            )
                    else:
                        result = compute_full_corr_matrix(
                            np.take(arr1[k, l, :, :, 0], idx, axis = 1),
                            np.take(arr2[k, l, :, :, 0], idx, axis = 1),
                            )

                    if same:
                        np.fill_diagonal(result, np.inf)
                        result = result[~np.isinf(result)]
                        result = result.reshape(arr1.shape[2], arr1.shape[2]-1)

                    output[i,j,k,l,:,:] = result

    if sep_conds:
        return output
    
    output = stat_func.r2z(output, 'pearson')
    output = np.mean(output, axis=2)
    output = stat_func.z2r(output, 'pearson')
    return output


def compute_full_rdm_corr_matrix(arr1, arr2):
    assert arr1[1].shape == arr2[1].shape, "RDMs must have the same number of samples"
    tril_idx = np.tril_indices(arr1.shape[1], k=-1)
    vec1 = np.array([rdm[tril_idx] for rdm in arr1])
    vec2 = np.array([rdm[tril_idx] for rdm in arr2])
    mat = compute_full_corr_matrix(vec1, vec2)
    return mat


def mapping_rdm(arr1: np.ndarray, 
                arr2: np.ndarray, 
                map_var: list, 
                same: bool,
                sep_conds: bool = False,
                n_bs: int = 1,
                seed: int = 42,
                ) -> np.ndarray:
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

    if same:
        output = np.zeros((n_bs, 2,
                           arr1.shape[0], arr1.shape[1], 
                           arr1.shape[2], arr1.shape[2] - 1
                           ))
    else:
        output = np.zeros((n_bs, 2,
                           arr1.shape[0], arr1.shape[1], 
                           arr1.shape[2], arr2.shape[2]
                           ))

    for i, (split1, split2) in enumerate(split_image_array(arr1, arr2, n_bs, seed)):
        for j, idx in enumerate([split1, split2]):
            for k in range(arr1.shape[0]):
                for l in range(arr1.shape[1]):
                    sub_arr1 = np.take(arr1[k, l, :, :, :], idx, axis = -2)
                    sub_arr1 = np.take(sub_arr1, idx, axis = -1)
                    sub_arr2 = np.take(arr2[k, l, :, :, :], idx, axis = -2)
                    sub_arr2 = np.take(sub_arr2, idx, axis = -1)
                    result = compute_full_rdm_corr_matrix(sub_arr1, sub_arr2)

                    if same:
                        np.fill_diagonal(result, np.inf)
                        result = result[~np.isinf(result)]
                        result = result.reshape(arr1.shape[2], arr1.shape[2]-1)

                    output[i,j,k,l,:,:] = result

    if sep_conds:
        return output
    
    output = stat_func.r2z(output, 'pearson')
    output = np.mean(output, axis=2)
    output = stat_func.z2r(output, 'pearson')
    return output


def retain_max_per_row_in_mat(arr: np.ndarray) -> np.ndarray:
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


def shuffle_image_order(arr: np.ndarray, seed: int = 42) -> np.ndarray:
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
    it = np.nditer(shuf_arr[..., 0], flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        last_dims = shuf_arr.shape[-1]
        perm = np.random.permutation(last_dims)
        shuf_arr[idx + (slice(None),)] = shuf_arr[idx + (perm,)]
        it.iternext()
    return shuf_arr


def center_to_zero(arr: np.ndarray) -> np.ndarray:
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





