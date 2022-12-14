from typing import Optional
import os
from multiprocessing import Pool, cpu_count
import glob
import re
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger('__main__')


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type, mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        print('std', df.std())
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
            if self.std is None:
                self.std = df.std()

            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y


class BaseData(object):

    def set_num_processes(self, n_proc):

        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())


class MimicData(BaseData):
    """
    Dataset class for Machine dataset.
    Attributes:
        all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """
    imputed_values_df = pd.read_csv('~/mvts_transformer/timeseries_transformer/resources/variable_ranges.csv',
                                    index_col=['LEVEL2'])

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, labels_path=None):

        self.set_num_processes(n_proc=n_proc)
        if labels_path == None:
            self.labels_path = "~/data/in-hospital-mortality/listfile.csv"
        else:
            self.labels_path = labels_path
        self.all_df, self.labels_df = self.load_all(root_dir, file_list=file_list, pattern=pattern)
        self.missing_mask = self.all_df.values == 0
        self.class_names = [0, 1]
        # print('missing_mask', self.missing_mask)
        self.all_df = self.all_df.set_index('ID')
        self.labels_df = self.labels_df.set_index('ID')

        self.all_IDs = self.all_df.index.unique()  # all sample (session) IDs
        # self.labels_df = self.labels_df[self.labels_df.index.isin(self.all_IDs.values)]
        self.max_seq_len = 100
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        self.feature_names = ["Diastolic blood pressure", "Systolic blood pressure", "Glascow coma scale total", "Respiratory rate", "Oxygen saturation", "Mean blood pressure", "Heart Rate", "Temperature", "Fraction inspired oxygen", 'Glucose', 'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale verbal response']
        self.feature_df = self.all_df[self.feature_names]

    def load_all(self, root_dir, file_list=None, pattern=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        Args:
            root_dir: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
        """
        # each file name corresponds to another date. Also tools (A, B) and others.

        labels_df = pd.read_csv(self.labels_path)

        input_paths = list(map(lambda path: "~/data/in-hospital-mortality/train/" + path, labels_df['stay'].values))

        labels_df = labels_df[['HADM_ID', 'y_true']]
        labels_df = labels_df.rename(columns={'HADM_ID': 'ID'})
        if self.n_proc > 1:
            # Load in parallel
            _n_proc = min(self.n_proc, len(input_paths))  # no more than file_names needed here
            logger.info("Loading {} datasets files using {} parallel processes ...".format(len(input_paths), _n_proc))
            with Pool(processes=_n_proc) as pool:
                all_df = pd.concat(pool.map(self.load_single, input_paths))
        else:  # read 1 file at a time
            all_df = pd.concat(self.load_single(path) for path in input_paths)

        return all_df, labels_df

    @staticmethod
    def load_single(filepath):
        df = MimicData.read_data(filepath)
        df = MimicData.select_columns(df)
        num_nan = df.isna().sum().sum()
        if num_nan > 0:
            df = df.interpolate(method='linear', limit_direction='forward', axis=0)
            df = df.fillna(df.mean())
            if df.isna().sum().sum() > 0:
                for col in df.columns:
                    if col == 'ID':
                        continue
                    df[col] = df[col].fillna(MimicData.imputed_values_df.loc[col, 'IMPUTE'])
            logger.warning("{} nan values in {} will be filled by linear interpolation and mean filling".format(num_nan, filepath))
        return df

    @staticmethod
    def read_data(filepath):
        """Reads a single .csv, which typically contains a day of datasets of various machine sessions.
        """
        df = pd.read_csv(filepath)
        return df

    @staticmethod
    def select_columns(df):
        """"""
        keep_cols = ["ID", "Diastolic blood pressure", "Systolic blood pressure", "Glascow coma scale total", "Respiratory rate", "Oxygen saturation", "Mean blood pressure", "Heart Rate", "Temperature", "Fraction inspired oxygen", 'Glucose', 'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale verbal response']
        df = df[keep_cols]

        return df

data_factory = {'mimic': MimicData}
