import os
from tqdm import tqdm
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

def parquetPCA(files_list, path="", n_components=10, itt_cap=None):
    """Takes a list of parquet files, reads it and redcues its size by applying a (vertical) PCA on each column.
    Returns a NumpPy array of dimmensions (len(files_list), *parquet.shape), i.e. creates a batch from the inputs list.
    
    - Be sure to import PCA from sklearn.decomposition (from sklearn.decomposition import PCA)
    """

    # Initialization of the batch array
    data = pd.read_parquet(os.path.join(path, files_list[0])).dropna(axis=0)
    batch = PCA(n_components).fit_transform(X=data.T).T[np.newaxis, ...]

    # Loop reading and concatenating every datapoint
    for parquet_file in tqdm(files_list[1:itt_cap]):
        data = pd.read_parquet(os.path.join(path, parquet_file)).dropna(axis=0)

        # If a sample is too short, then it is padded with zeros (unlikely, more frequent as n_components increases)
        if data.shape[0] < n_components:
            pad_width = [(n_components - data.shape[0], 0), (0, 0)]
            data = np.pad(data, pad_width, mode="edge")

        data_new = PCA(n_components).fit_transform(X=data.T).T[np.newaxis, ...]
        batch = np.concatenate([batch, data_new], axis=0) # Concatenates alongside batch axis

    print("parquetPCA: output shape is", batch.shape)
    return batch


# eegs_batch = parquetPCA(files_list=train_eegs_list, path=TRAIN_EEGS_PATH, itt_cap=2, n_components=20)
# spect_batch = parquetPCA(files_list=train_spect_list, path=TRAIN_SPECT_PATH, itt_cap=2, n_components=20)
