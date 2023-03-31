"""
File contests:
Modified utils.py file from original code skeleton

Added a few helper functions for preparing data, fixing invalid XGBoost configurations, and performing mappings like data_id => task_id

"""

import os
import openml
import numpy as np
import pandas as pd
import statistics
from typing import Dict, List, Tuple

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

"""
Files in data folder:
1.features.csv - original meta features file, 94 rows - each describes a single dataset !!! NO FEATURES FOR TEST DATASETS
2.xgboost_meta_data.csv - original HPC data - 3,3m rows - (dataset,HPC,replication)->(score,training time) 
3.average_performance.csv - average performance for each (dataset,HPC) pair - 687.101 rows -  (dataset,HPC)->(avg_score, avg_time)
4.joint_data.csv - average performance expanded with dataset meta features - 687.101 rows - (dataset,HPC)->(avg_score, avg_time, dataset_metafeatures)
5.baseline_performance.csv - default_config performance on each of the TEST tasks (datasets) - score and train_time
6.The rest of the CSV files compare the baseline performance with that of configurations recommended by one of the regression models used
"""

"""
Tasks with ids: 146825, 168332 
were removed due to their training time on the default configuration
eg. 146825 did not finish after an hour with 4 threads and 100% CPU usage
"""
test_ids = [
    16, 22, 31, 2074, 2079, 3493, 3907, 3913, 9950, 9952, 9971, 10106, 14954, 14970, 146212, 167119, 167125, 168336
]

meta_feature_names = [
    "data_id", "name", "status", "MajorityClassSize", "MaxNominalAttDistinctValues",
    "MinorityClassSize", "NumberOfClasses", "NumberOfFeatures", "NumberOfInstances",
    "NumberOfInstancesWithMissingValues", "NumberOfMissingValues", "NumberOfNumericFeatures",
    "NumberOfSymbolicFeatures"
]

default_config = {
    "n_estimators": 464,  # num_rounds
    "eta": 0.0082,
    "subsample": 0.982,
    "max_depth": 11,
    "min_child_weight": 3.30,
    "colsample_bytree": 0.975,
    "colsample_bylevel": 0.9,
    "lambda": 0.06068,
    "alpha": 0.00235,
    "gamma": 0
}

# Dictionary or data_id : task_id used to add task_id column to training data
dataset_to_task = {3: 3, 6: 6, 11: 11, 12: 12, 14: 14, 15: 15, 18: 18, 23: 23, 24: 24, 28: 28, 29: 29,
                   32: 32, 37: 37, 38: 3021, 42: 41, 44: 43, 46: 45, 50: 49, 54: 53, 60: 58, 151: 219, 181: 2073,
                   300: 3481, 307: 3022,
                   312: 3485, 375: 3510, 377: 3512, 458: 3549, 469: 3560, 470: 3561, 554: 3573, 1040: 3893, 1049: 3902,
                   1050: 3903,
                   1053: 3904, 1067: 3917, 1068: 3918, 1111: 3945, 1457: 10090, 1461: 14965, 1462: 10093, 1464: 10101,
                   1468: 9981, 1475: 9985, 1476: 9986, 1479: 9970, 1485: 9976, 1486: 9977, 1487: 9978, 1493: 9956,
                   1494: 9957, 1497: 9960, 1501: 9964, 1510: 9946, 1590: 7592, 4134: 9910, 4135: 34539, 4534: 14952,
                   4538: 14969, 4541: 168339, 23381: 125920, 23512: 146606, 23517: 167120, 40496: 125921, 40498: 145681,
                   40499: 125922, 40536: 146607, 40668: 146195, 40670: 167140, 40701: 167141, 40900: 168759,
                   40923: 167121, 40927: 167124,
                   40966: 146800, 40975: 146821, 40979: 146824, 40981: 146818, 40982: 146817, 40983: 146820,
                   40984: 146822,
                   40994: 146819, 41142: 168765, 41143: 168764, 41146: 168761, 41150: 168335, 41156: 168767,
                   41157: 168768,
                   41159: 168337, 41161: 168338, 41163: 168770, 41164: 168760, 41166: 168331, 41168: 168330,
                   41169: 168329}
"""
Names of hyperparameters in the CSVs
"""
hyperparameters_data = ['num_round', 'eta', 'gamma', 'lambda', 'alpha', 'subsample',
                        'max_depth', 'min_child_weight', 'colsample_bytree',
                        'colsample_bylevel']

"""
All dataset metafeatures used during training - features such as data_id, name, version and status are uninformative
or unique which leads to noise and overfitting respectively, so they are excluded

"""
training_meta_features: List[str] = ['MajorityClassSize', 'MaxNominalAttDistinctValues', 'MinorityClassSize',
                                     'NumberOfClasses', 'NumberOfFeatures', 'NumberOfInstances',
                                     'NumberOfInstancesWithMissingValues', 'NumberOfMissingValues',
                                     'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures']

"""
Converts the predictions of the regression model (typically continuous) into valid configurations for XGBoost
"""


def make_valid_config(config: dict) -> dict:
    valid_confg = config.copy()
    # int and >=0, Also we have to change the name
    if 'num_round' in valid_confg.keys():
        valid_confg['n_estimators'] = max(0, int(valid_confg['num_round']))
        del (valid_confg['num_round'])
    else:
        valid_confg['n_estimators'] = max(0, int(valid_confg['n_estimators']))
    # float and [0,1]
    valid_confg['eta'] = min(max(0, valid_confg['eta']), 1)
    # float and (0,1]
    valid_confg['subsample'] = min(max(10e-2, valid_confg['subsample']), 1)
    # int and >=0
    valid_confg['max_depth'] = max(0, int(valid_confg['max_depth']))
    # float and >=0
    valid_confg['min_child_weight'] = max(0, valid_confg['min_child_weight'])
    # floats and (0,1]
    valid_confg['colsample_bytree'] = min(max(10e-2, valid_confg['colsample_bytree']), 1)
    valid_confg['colsample_bylevel'] = min(max(10e-2, valid_confg['colsample_bylevel']), 1)
    # float and >=0
    valid_confg['lambda'] = max(0, valid_confg['lambda'])
    valid_confg['alpha'] = max(0, valid_confg['alpha'])
    valid_confg['gamma'] = max(0, valid_confg['gamma'])

    return valid_confg


"""
Returns the data_id to task_id dictionary
"""


def get_dataset_to_task() -> dict:
    return dataset_to_task


"""
Same as previous method just flips the mapping
"""


def get_task_to_dataset() -> dict:
    task_to_dataset = {v: k for k, v in dataset_to_task.items()}
    return task_to_dataset


# Old function - skeleton
def get_hyperparameter_list() -> List[str]:
    return list(default_config.keys())


# Old function - skeleton
def get_metafeature_list() -> List[str]:
    return meta_feature_names


# Old function - skeleton
def load_data_from_path(path_to_files: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the meta files from disk and returns them as data frames"""
    files_to_load = dict(
        features="../features.csv",
        meta_features="../xgboost_meta_data.csv"
    )
    meta_features = pd.read_csv(os.path.join(path_to_files, files_to_load["features"]))
    meta_data = pd.read_csv(os.path.join(path_to_files, files_to_load["meta_features"]))
    return meta_features, meta_data


# Old function - skeleton
def _get_preprocessor(categoricals, continuous):
    """Preprocessing"""
    preprocessor = make_pipeline(
        ColumnTransformer([
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore")
                ),
                categoricals.tolist(),
            ),
            (
                "cont",
                make_pipeline(
                    SimpleImputer(strategy="median")
                ),
                continuous.tolist(),
            )
        ])
    )
    return preprocessor


# Old function - skeleton
def get_task_metafeatures(task_id: int, meta_feature_names: List[str]) -> Dict:
    """Get meta features from an OpenML task based on its task id"""
    task = openml.tasks.get_task(task_id)
    features = openml.datasets.list_datasets(data_id=[task.dataset_id])[task.dataset_id]
    features["data_id"] = features["did"]

    for feature in set(features.keys()) - set(meta_feature_names):
        features.pop(feature)

    return features


# Old function - skeleton
def _convert_labels(labels):
    """Converts boolean labels (if exists) to strings"""
    label_types = list(map(lambda x: isinstance(x, bool), labels))
    if np.all(label_types):
        _labels = list(map(lambda x: str(x), labels))
        if isinstance(labels, pd.Series):
            labels = pd.Series(_labels, index=labels.index)
        elif isinstance(labels, np.array):
            labels = np.array(labels)
    return labels


# Old function - skeleton
def load_test_data(task_id: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetches data from OpenML, converts data types, to yield train-test numpy arrays"""
    task = openml.tasks.get_task(task_id, download_data=False)
    nclasses = len(task.class_labels)
    dataset = openml.datasets.get_dataset(task.dataset_id, download_data=False)
    X, y, categorical_ind, feature_names = dataset.get_data(target=task.target_name, dataset_format="dataframe")

    categorical_ind = np.array(categorical_ind)
    (cat_idx,) = np.where(categorical_ind)
    (cont_idx,) = np.where(~categorical_ind)

    # splitting dataset into train and test (10% test)
    # train-test split is fixed for a task and its associated dataset (from OpenML)
    train_idx, test_idx = task.get_train_test_split_indices()  # we only use the first of the 10 CV folds

    train_x = X.iloc[train_idx]
    train_y = y.iloc[train_idx]
    test_x = X.iloc[test_idx]
    test_y = y.iloc[test_idx]

    preprocessor = _get_preprocessor(cat_idx, cont_idx)

    # preprocessor fit only on the training set
    train_x = preprocessor.fit_transform(train_x)
    test_x = preprocessor.transform(test_x)

    # converting bool labels
    train_y = _convert_labels(train_y)
    test_y = _convert_labels(test_y)

    # encoding labels
    le = LabelEncoder()
    le.fit(np.unique(train_y))
    train_y = le.transform(train_y)
    test_y = le.transform(test_y)

    return train_x, train_y, test_x, test_y, nclasses


"""
For each of the 94 training tasks returns the HPC that achieved the best average AUC across its runs
"""


def get_best_config_per_task(n_best: int = 1) -> pd.DataFrame:
    joint_data = pd.read_csv('./data/joint_data.csv')
    # used to be just .first() => switched to .head(n_best)
    best_configs = joint_data.sort_values(['avg_auc'], ascending=False).groupby('data_id').head(n_best).reset_index()
    best_configs = best_configs.drop(columns=['index'])
    return best_configs


"""
For getting test dataset metafeatures
    Later mostly not used, instead we just load from ./data/test_features.csv
"""


def get_test_metafeatures(address='./data/test_features.csv') -> pd.DataFrame:
    if os.path.isfile(address):
        features = pd.read_csv(address)
        return features
    else:
        features = pd.DataFrame(columns=meta_feature_names.append('task_id'))
        for task_id in test_ids:
            feature_dict = get_task_metafeatures(task_id, meta_feature_names)
            feature_dict.update({'task_id': task_id})
            features = features.append(feature_dict, ignore_index=True)
        features.to_csv(address)
        return features


"""
    Prepares the training and test data
"""


def get_all_metafeatures(impute=True, best_configs=True) -> (pd.DataFrame, pd.DataFrame):
    """
    Train data - Train datasets with their best configurations according to their average AUC
    Test data - Features of the Test tasks that need to be used to predict the best XGBoost config
    For both we impute the missing values with a median (reason: long tail distribution of the single column with Nans)
    """
    if not best_configs:
        train_metadata = pd.read_csv('./data/features.csv')
    else:
        train_metadata = get_best_config_per_task()

    test_metadata = pd.read_csv('./data/test_features.csv')
    data_to_task_id = get_dataset_to_task()
    # Adding task_id values from the xgboost data (features.csv doesn't contain the task_id)
    train_metadata['task_id'] = train_metadata['data_id'].map(data_to_task_id)

    if 'version' in train_metadata.columns:
        train_metadata = train_metadata.drop('version', axis=1)

    """
    Imputing NaNs with the median value - reason for this is that MaxNominalAttDistinctValues was the only column with 
    NaNs and it had a long tail so taking the average would not make sense
    """
    if not impute:
        return train_metadata, test_metadata
    else:
        # Only column with missing values
        nan_column = 'MaxNominalAttDistinctValues'
        # Finding median across both train and test datasets
        train_not_nan = train_metadata[~train_metadata[nan_column].isna()][nan_column].tolist()
        test_not_nan = test_metadata[~test_metadata[nan_column].isna()][nan_column].tolist()
        nan_column_median = int(statistics.median(train_not_nan + test_not_nan))

        # Replacing NaNs with median
        train_metadata[nan_column] = train_metadata[nan_column].fillna(nan_column_median)
        test_metadata[nan_column] = test_metadata[nan_column].fillna(nan_column_median)
        return train_metadata, test_metadata


"""
    Retrieves data about average performance of different configurations on each of the training tasks
"""


def get_average_performance(time_limit: int = 600, per_task=False):
    avg_perf = pd.read_csv('./data/average_performance.csv')
    avg_perf = avg_perf[avg_perf['avg_time'] <= time_limit]
    task_dict = get_dataset_to_task()
    avg_perf['task_id'] = avg_perf['data_id'].map(task_dict)
    avg_perf = avg_perf.drop(['index'], axis=1, errors='ignore')
    if not per_task:
        return avg_perf
    else:
        """Here we create a dictionary of task_id -> performance data"""
        task_ids = avg_perf['task_id'].unique()
        # Create a dataframe for each of the 94 training tasks
        training_data_per_task = {task_id: pd.DataFrame() for task_id in task_ids}

        # Select the corresponding rows
        for key in training_data_per_task.keys():
            training_data_per_task[key] = avg_perf[:][avg_perf['task_id'] == key]
        return training_data_per_task


def create_average_performance_data(address='./data/average_performance.csv'):
    meta_df = pd.read_csv("./data/features.csv")
    runs_df = pd.read_csv("./data/xgboost_meta_data.csv")
    grouped_runs_all = runs_df.groupby(
        ['data_id', 'num_round', 'eta', 'gamma', 'lambda', 'alpha', 'subsample', 'max_depth',
         'min_child_weight', 'colsample_bytree', 'colsample_bylevel'])
    # %%

    average_performance = grouped_runs_all['auc', 'timetrain'].mean().reset_index()
    average_performance = average_performance.rename(columns={"timetrain": "avg_time", "auc": "avg_auc"})
    average_performance
    print("Average performance data \n")
    print(average_performance)
    # Save configuration info
    average_performance.to_csv(address)


def create_joint_data(address='./data/joint_data.csv'):
    average_performance = pd.read_csv('./data/average_performance.csv')
    average_performance
    features_data = pd.read_csv('./data/features.csv')
    features_data

    joint_data = average_performance.merge(features_data, on='data_id', how='inner', suffixes=('_features', '_meta'))
    joint_data = joint_data.drop(['index', 'version', 'status', ], axis=1)
    joint_data.to_csv(address, index=False)


"""
For calculating standardised distances between datasets based on metafeatures
NOT USED - replaced with different function for joint and taskwise regression
"""
# def metafeature_distances(new_datasets: pd.DataFrame = None, training_datasets: pd.DataFrame = None) -> pd.DataFrame:
#     columns = training_meta_features.copy()
#     columns.insert(0, 'data_id')
#
#     training_data = training_datasets[columns]
#     train_ids = training_data['data_id'].values.tolist()
#     train_ids.sort()
#
#     new_data = new_datasets[columns]
#     new_ids = new_data['data_id'].values.tolist()
#     new_ids.sort()
#
#     full_data = training_data.append(new_data)
#     print(full_data)
#
#     distances = pd.DataFrame(columns=['new_data_id', 'train_data_id', 'distance'])
#     # for new_id in new_ids:
#     #     features_new = full_data.loc[full_data['data_id'] == new_id].head(1)[training_meta_features]
#     #     for train_id in train_ids:
#     #         features_train = (full_data.loc[full_data['data_id'] == train_id].head(1))[training_meta_features]
#     #         dist = np.linalg.norm(features_train[training_meta_features].values - features_new[training_meta_features].values, axis=1)[0]
#     #         new_row = {
#     #             'new_data_id':new_id,
#     #             'train_data_id':train_id,
#     #             'distance':dist
#     #         }
#     #         distances = distances.append(new_row, ignore_index=True)
#
#     return distances
